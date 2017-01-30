import numpy              as np
import random             as rd
import csv
import cv2

from keras.models                       import Sequential
from keras.layers                       import Flatten, Dense, Activation, Dropout, MaxPooling2D, Convolution2D
from keras.layers.advanced_activations  import ELU
from keras.callbacks                    import EarlyStopping

# Define the expected size for the network
NETWORK_IMAGE_SIZE = (64, 64)


def extract_csv_data(csv_file):
    '''
        Extract the information from the csv file provided
        
        Expecting format :
        
        No header record
        Columns
        0 - Center image file
        1 - Left image file
        2 - Right image file
        3 - Steering angle
    '''
    csv_data = []
    
    csvfh = open(csv_file, newline='')
    csvreader = csv.reader(csvfh, delimiter=',')
    
    for row in csvreader:
        data = {}
        # Add the images
        data['left']   = row[1]
        data['center'] = row[0]
        data['right']  = row[2]
        # Add the steering angle
        data['steering_angle'] = float(row[3])
        
        csv_data.append(data)
    csvfh.close()
    
    return csv_data

def network_model():
    '''
        Define our network architecure using Keras
    '''
    
    model = Sequential()
    
    # Conv Stage 1
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(NETWORK_IMAGE_SIZE[0], NETWORK_IMAGE_SIZE[1],3)))
    model.add(ELU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    
    # Conv Stage 2
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(ELU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    
    model.add(Dropout(0.5))
    
    # Conv Stage 3
    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(ELU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    
    # Conv Stage 4
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(ELU(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    
    model.add(Dropout(0.5))

    model.add(Flatten())
    
    # FC Stage 1
    model.add(Dense(256))
    model.add(ELU(0.3))
    
    # FC Stage 2
    model.add(Dense(128))
    model.add(ELU(0.3))
    
    model.add(Dropout(0.5))
    
    # FC Stage 3
    model.add(Dense(64))
    model.add(ELU(0.3))

    # FC output
    model.add(Dense(1))

    return model
    
def train_model(model, train_data, batch_size, epochs, samples_by_epoch):
    '''
        Train the model using a generator
        The validation set size is a fourth of the training set size.
    '''

    # Define optimizer, loss 
    model.compile(optimizer='adam', loss='mse')

    # Define our early stopping condition.
    # We use the validation loss as indicator, no tolerance
    # and wait max 2 epochs
    early_stopping = EarlyStopping( monitor='val_loss'
                                   ,min_delta=0
                                   ,patience=2
                                  )
                           
    # Effectively train our model using a generator
    model.fit_generator(  generate_network_training_input(train_data, batch_size)
                        , samples_per_epoch=(samples_by_epoch // batch_size) * batch_size
                        , nb_epoch=epochs
                        , validation_data=generate_network_training_input(train_data, batch_size)
                        , nb_val_samples=( (samples_by_epoch // 4) // batch_size) * batch_size
                        , callbacks=[early_stopping]
                       )
    
    
def save_model(model, filename_prefix='model'):
    '''
        Just save the model and the weights, so we can reuse it.
    '''
    # Save the weights
    model.save_weights('./' + filename_prefix  + '.h5')

    # and the model
    with open('./' + filename_prefix + '.json', 'w') as f:
        f.write(model.to_json())

        
def crop_car_hood(img):
    '''
        Crop the bottom of the image to remove car hood
    '''
    return img[:135,:,:]
    
    
def preprocess_image(img, do_crop_car_hood=True):
    '''
        Preprocess the image to prepare it for the network
        We are removing the top to remove the sky
        We are removing the bottom to remove the car hood if requested
        We are normalizing the image
        We are resizing the image for the network input
    '''
    
    # Crop the bottom to remove the car hood
    if do_crop_car_hood:
        img = crop_car_hood(img)

    # Crop the image to remove upper part of it where the sky belong
    img = img[50:,:,:]
    
    # Normalize and center at origin
    img = img / 255
    img -= np.mean(img)
    
    # Resize it
    img = cv2.resize(img, NETWORK_IMAGE_SIZE , cv2.INTER_CUBIC)
    
    return img

def random_luminosity_on_image(img):
    '''
        Modify randomly the luminosity of the image
        Image has to be in RGB format
        Use a factor between .25 to 1.25
    '''
    
    # Convert to HSV to simplify the operation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Update the channel V to update the luminosity
    hsv[:,:,2] = hsv[:,:,2] * rd.uniform(0.25, 1.25)
    
    # Convert back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_shadow_on_image(img, mask):
    '''
        Add the shadow that the mask contains on the image
        Mask must have the same shape as image
        The level of shadow is randomized
    '''
    assert img.shape == mask.shape, 'Image and mask must have the same shape'
    
    # Create a copy of our image to apply the mask
    # Add a random luminosity so we get different shadow level
    img_copy = random_luminosity_on_image(np.copy(img))
        
    # Get the masked image from the copy ( ie our shadow )
    img_shadow = cv2.bitwise_and(img_copy, mask)

    # Get the image with the shadow part removed
    img_neg = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    
    # return the Merge image
    return img_shadow + img_neg

def add_random_triangle_on_image(img):
    '''
        Add a random triangle on the image
    '''
    height, width = img.shape[0:2]
    
    # Generate 3 random points
    points = np.array([[ [rd.randint(0,width-1), rd.randint(0,height-1)]
                        ,[rd.randint(0,width-1), rd.randint(0,height-1)]
                        ,[rd.randint(0,width-1), rd.randint(0,height-1)]
                      ]]
                     )
    
    # Get a mask images
    mask = np.zeros_like(img)
    
    # Draw our triangle on mask
    mask = cv2.fillPoly(mask, points, [255,255,255])
    
    return add_shadow_on_image(img, mask)

def add_random_circle_on_image(img):
    '''
        Add a random circle on the image
    '''
    height, width = img.shape[0:2]
    
    # Generate random center, radius
    center = (rd.randint(0,width-1), rd.randint(0,height-1))
    radius = rd.randint(0, height // 6)
    
    # Get a mask images
    mask = np.zeros_like(img)
    
    # Draw our circle on mask
    mask = cv2.circle(mask, center, radius, [255,255,255], thickness=-1)
    
    return add_shadow_on_image(img, mask)

def add_random_polygon_on_image(img):
    '''
        Add a random polygon of 4 edges on the image
    '''
    height, width = img.shape[0:2]
    
    # Generate 4 random points
    points = np.array([[ [rd.randint(0,width//2), rd.randint(0,height//2)]
                        ,[rd.randint(width//2,width-1), rd.randint(0,height//2)]
                        ,[rd.randint(width//2,width-1), rd.randint(height//2,height-1)]
                        ,[rd.randint(0,width//2), rd.randint(height//2,height-1)]
                      ]]
                     )
    
    # Get a mask images
    mask = np.zeros_like(img)
    
    # Draw our polygon on mask
    mask = cv2.fillPoly(mask, points, [255,255,255])
    
    return add_shadow_on_image(img, mask)

def add_random_occlusion_on_image(img):
    '''
        Add a random polygon of 4 edges on the image
        fixing 2 points on either top_left/top_right
        and bottom_left/bottom_right
    '''
    height, width = img.shape[0:2]
    
    # Generate 4 random points
    # either on left side or right side
    if rd.random() < 0.5:
        points = np.array([[ [0, 0]
                            ,[rd.randint(width // 4,width*3 // 4 ), 0]
                            ,[rd.randint(width // 8,width // 4 ), height-1]
                            ,[0, height-1]
                          ]]
                     )
    else:
        points = np.array([[ [rd.randint(width // 4,width*3 // 4), 0]
                            ,[width-1, 0]
                            ,[width-1, height-1]
                            ,[rd.randint(width*3 // 4,width *7 // 8), height-1]
                          ]]
                     )
    
    
    # Get a mask images
    mask = np.zeros_like(img)
    
    # Draw our polygon on mask
    mask = cv2.fillPoly(mask, points, [255,255,255])
    
    return add_shadow_on_image(add_shadow_on_image(img, mask), mask)

def random_shadow_objects_on_image(img):
    '''
        Generate random object on the image
        To simulate occlusion and shadow cast
    '''
    
    # Generate up to 5 objects
    n_object = rd.randint(1,5)
    
    for n in range(n_object):
        object_type = rd.randint(0,2)
        
        if   object_type == 0:  img = add_random_triangle_on_image(img)
        elif object_type == 1:  img = add_random_circle_on_image(img)
        elif object_type == 2:  img = add_random_polygon_on_image(img)
    
    if rd.random() < 0.5:
        img = add_random_occlusion_on_image(img)
    
    return img


def translate_data(img, steer, ratio_x=0.3, ratio_y=0.3):
    '''
        Translate our image vertically to simulate slope street
        and horizontally to generate recovery data and extreme case
        When translated horizontally we adjust the steering angle in
        proportion of the translation
        
        The ratio parameters give us the max translation in the axis
        based on image size.
    '''
    
    height, width = img.shape[0:2]
        
    # Get our range
    tx_range = width  * ratio_x
    ty_range = height * ratio_y
    
    # Get a ramdomized translation
    tx = int(rd.uniform(-1.0, 1.0) * tx_range)
    ty = int(rd.uniform(-1.0, 1.0) * ty_range)
    
    # Translation matrix
    M = np.array([ [ 1.0, 0.0, tx]
                  ,[ 0.0, 1.0, ty]
                 ])
    
    # Apply the translation
    img = cv2.warpAffine(img, M, (width, height))

    # Adjust the steering angle, we will use a 0.006 angle by pixel
    # Use this to get 1 if we shift half image 1 / (320/2)
    steer += tx * 0.006
    # Keep it in range -1,1
    steer = min(1.0, max(-1.0, steer))
    
    return img, steer

def generate_network_training_input(train_data, batch_size):
    '''
        Generator to feed the network during training with augmented data
        on the fly. Just the batch size is kept in memory.
        We always generate a new set image/steering angle from
        the training data provided.
        
        Different techniques are used to generate those images
        Crop, translation, luminosity, occlusion, shadows...
        
        Also it tries to keep the dataset uniform using a retry process
        to keep the standard deviation below 5
    '''
    while 1:
        
        batch_img   = np.zeros((batch_size, NETWORK_IMAGE_SIZE[0], NETWORK_IMAGE_SIZE[1], 3))
        batch_steer = np.zeros((batch_size, 1))
        train_steer_bins = np.zeros( int(2.0 / 0.04) + 1 )

        # Fill up our batch with new data
        for batch_idx in range(batch_size):
            
            # A try loop to uniform our train dataset to avoid network bias
            for try_i in range(100):
                # Get a sample
                train_i = rd.randint(0, len(train_data)-1)
                
                # Extract the image
                img = cv2.cvtColor(cv2.imread(train_data[train_i]['center']), cv2.COLOR_BGR2RGB)
                # Extract steering
                steering_angle = train_data[train_i]['steering_angle']
                
                # Crop bottom to remove the car Hood
                img = crop_car_hood(img)
                
                # Add some shadow/occlusion
                img = random_shadow_objects_on_image(img)
                    
                # Fluctuate the luminosity of the image
                img = random_luminosity_on_image(img)
                
                # Generate a translated image
                img, steering_angle = translate_data(img, steering_angle)
                
                # Flip the image (vertival flip) as well as the steering angle
                if rd.random() > 0.5 :
                    img = cv2.flip(img, 1)
                    steering_angle *= -1.0
                    
                # Evaluate if this is a good candidate
                # Get the bin to add our count
                steer_bin = int( (steering_angle + 1.0) / 0.04 )
                train_steer_bins[steer_bin] += 1
                
                # Do not retry if not enough samples
                mean = np.mean(train_steer_bins)
                if mean < 30: break
                
                # Evaluate the stddev to make a decision
                stdev = np.stdev(train_steer_bins)
                if stdev <= 5.0: break
                
                # We are not a good candidate, so remove our count
                train_steer_bins[steer_bin] -= 1


            # Preprocess the image ( crop, normalize, resize ... )
            img = preprocess_image(img, do_crop_car_hood=False)
            
            batch_img[batch_idx]   = img
            batch_steer[batch_idx] = steering_angle
           
        yield (batch_img, batch_steer)

