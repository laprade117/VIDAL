import cv2
import numpy as np

def rescale_image(image, res, target_res=6, interpolation=cv2.INTER_CUBIC):
    
    scale_factor = target_res / res
    
    new_shape = np.round(np.array(image.shape[:2]) * scale_factor).astype(int)
    new_shape = (new_shape[1], new_shape[0])
    
    rescaled_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
    
    return rescaled_image

def srgb_to_linear(srgb): 
    # https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color/56678483#56678483
    srgb = srgb / 255.0
    
    upper = ((srgb + 0.055) / 1.055)**2.4
    lower = srgb / 12.92
    linear = np.where(srgb > 0.04045, upper, lower)
    
    linear = np.clip(np.round(linear * 255), 0, 255).astype('uint8')
    return linear

def linear_to_srgb(linear):
    linear = linear / 255.0
    
    upper = 1.055 * linear**(1.0 / 2.4) - 0.055
    lower = 12.92 * linear
    srgb = np.where(linear > 0.0031308, upper, lower)
        
    srgb = np.clip(np.round(srgb * 255), 0, 255).astype('uint8')
    return srgb

def auto_white_balance(image, clipping=0.0005):
    
    # https://stackoverflow.com/questions/48268068/how-do-i-do-the-equivalent-of-gimps-colors-auto-white-balance-in-python-fu
    
    balanced_image = np.zeros(image.shape)
    
    for i in range(3):
        hist = np.histogram(image[..., i].ravel(), 256, (0, 256))[0]

        clipping_range = np.where(hist > (hist.sum() * clipping))

        c_min = np.min(clipping_range)
        c_max = np.max(clipping_range)

        balanced_image[...,i] = np.clip(image[...,i], c_min, c_max)
        balanced_image[...,i] = (balanced_image[...,i] - c_min) / (c_max - c_min)
    
    balanced_image = (balanced_image * 255).astype('uint8')
    return balanced_image

def find_black_square(image, multiscale_factor=0.95, t1=200, t2=200, return_template=False):
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Define black square template
    template = np.pad(np.ones((16,16), dtype='uint8') * 255, ((2,2),(2,2)))
    template = np.pad(template, ((2,2),(2,2)), constant_values=255)
    template = cv2.resize(template, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST) 
    edge_template = cv2.Canny(template, 100, 200)
    
    # Do multiscale template matching on edges
    highest_val = 0
    coordinates = None
    for i in range(30):
        
        # Scale image and find edges
        current_scale = multiscale_factor**i
        resized_image = cv2.resize(gray_image,
                                   (0,0),
                                   fx=current_scale,
                                   fy=current_scale,
                                   interpolation=cv2.INTER_NEAREST)
        edge_image = cv2.Canny(resized_image, t1, t2)

        # Search for template
        result = cv2.matchTemplate(edge_image, edge_template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update best match
        if max_val > highest_val:
            highest_val = max_val
            coordinates = np.round(np.array(max_loc) / current_scale).astype(int)
            bbox_size = np.round(template.shape[0] / current_scale).astype(int)
    
    black_square = image[coordinates[1]:coordinates[1] + bbox_size,
                         coordinates[0]:coordinates[0] + bbox_size,:]
    scaled_template = cv2.resize(template, (bbox_size, bbox_size), interpolation=cv2.INTER_NEAREST)
    
    if return_template:
        return black_square, scaled_template
    else:
        return black_square
        
def find_all_black_squares(image, multiscale_factor=0.95, t1=200, t2=200):
    
    split_value = (np.array(image.shape) / 2).astype(int)
    
    black_squares = []

    for i in range(2):
        start_x = i * split_value[0]
        end_x = i * split_value[0] + split_value[0]
        for j in range(2):
            start_y = j * split_value[1]
            end_y = j * split_value[1] + split_value[1]
            
            quadrant = image[start_x:end_x,start_y:end_y,:]
            black_square = find_black_square(quadrant, multiscale_factor=multiscale_factor, t1=t1, t2=t2)
            black_squares.append(black_square)
            
    return black_squares

def estimate_white_point(image, return_area=False):
    
    # Threshold grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (cv2.threshold(gray_image, 0.5*255, 255, cv2.THRESH_BINARY)[1] / 255).astype('uint8')
    
    # Compute white area and white point
    white_area = np.sum(mask)
    white_point = np.sum(image * mask[:,:,None], axis=(0,1)) / white_area
    
    if return_area:
        return white_point, white_area
    else:
        return white_point
    
def estimate_black_point(image, return_area=False):
    
    # Threshold grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = 1 - (cv2.threshold(gray_image, 0.25*255, 255, cv2.THRESH_BINARY)[1] / 255).astype('uint8')
    
    # Compute black area and black point
    black_area = np.sum(mask)
    black_point = np.sum(image * mask[:,:,None], axis=(0,1)) / black_area
    
    if return_area:
        return black_point, black_area
    else:
        return black_point
    
def estimate_pixel_resolution(image, inner_area=10**2, return_pixel_area=False):

    # Threshold grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1] / 255).astype('uint8')

    # Connected components
    # from scipy.ndimage import label, generate_binary_structure
    # labeled_array, num_features = label(mask, structure=generate_binary_structure(2,1))
    num_features, labeled_array = cv2.connectedComponents(mask)

    # Find and mask largest object
    object_mask = (labeled_array == np.argmax([np.sum(labeled_array == i) for i in range(num_features)]))

    # Compute pixel area and resolution
    pixel_area = np.sum(object_mask)
    pixel_resolution = np.sqrt(inner_area) / np.sqrt(pixel_area)

    if return_pixel_area:
        return pixel_resolution, pixel_area
    else:
        return pixel_resolution

def compute_square_info(black_squares, inner_area=10**2):
    
    total_inner_area = 0
    total_pixel_area = 0

    total_white_area = 0
    total_white_point = np.zeros(3)

    total_black_area = 0
    total_black_point = np.zeros(3)

    for i in range(len(black_squares)):

        pixel_resolution, pixel_area = estimate_pixel_resolution(black_squares[i],
                                                                 inner_area=inner_area,
                                                                 return_pixel_area=True)
        total_inner_area += inner_area
        total_pixel_area += pixel_area

        white_point, white_area = estimate_white_point(black_squares[i],
                                                       return_area=True) 
        black_point, black_area = estimate_black_point(black_squares[i],
                                                       return_area=True)
        total_white_point += white_point * white_area
        total_white_area += white_area

        total_black_point += black_point * black_area
        total_black_area += black_area

    white_point = total_white_point / total_white_area           
    black_point = total_black_point / total_black_area

    pixel_resolution = 1 / (np.sqrt(total_inner_area) / np.sqrt(total_pixel_area))
    
    return white_point, black_point, pixel_resolution

def white_balance(image, white_point, black_point):
    
    image = image.astype(float)

    lum = np.mean(white_point)
    image[...,0] = (image[...,0] - black_point[0]) * lum / (white_point[0] - black_point[0])
    image[...,1] = (image[...,1] - black_point[1]) * lum / (white_point[1] - black_point[1])
    image[...,2] = (image[...,2] - black_point[2]) * lum / (white_point[2] - black_point[2])

    image = np.round(np.clip(image, 0, 255)).astype('uint8')
    
    return image

def rescale_image(image, res, target_res=6, interpolation=cv2.INTER_CUBIC):
    
    scale_factor = target_res / res
    
    new_shape = np.round(np.array(image.shape[:2]) * scale_factor).astype(int)
    new_shape = (new_shape[1], new_shape[0])
    
    rescaled_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
    
    return rescaled_image

def preprocess_image(image, inner_area=10**2, target_res=5):

    # Convert to linear RGB color space
    image = srgb_to_linear(image)

    # Apply automatic white balancing
    image = auto_white_balance(image)

    # Apply template white balancing
    black_squares = find_all_black_squares(image)
    white_point, black_point, pixel_resolution = compute_square_info(black_squares, inner_area=inner_area)
    image = white_balance(image, white_point, black_point)

    # Convert back to gamma RGB color space
    image = linear_to_srgb(image)

    # Rescale image to have a resolution of 5 pixels per mm
    image = rescale_image(image, pixel_resolution, target_res=target_res, interpolation=cv2.INTER_CUBIC)

    return image