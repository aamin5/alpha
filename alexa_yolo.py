import numpy as np
import time
import cv2


def AngelEye():
        # load the coco class lables
        LABELS = open("coco.names").read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")



        # loading yolo object detctor based on coco data set 
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

        # load input image
        videoCaptureObject = cv2.VideoCapture(0)

        ret,frame = videoCaptureObject.read()
        cv2.imwrite("input/NewPicture.jpg",frame)
           
        videoCaptureObject.release()
        cv2.destroyAllWindows()

        image = cv2.imread("input/NewPicture.jpg")
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        centers = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.5:
                                # scale the bounding box coordinates back relative to the
                                # size of the image, keeping in mind that YOLO actually
                                # returns the center (x, y)-coordinates of the bounding
                                # box followed by the boxes' width and height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences,
                                # and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                centers.append((centerX, centerY))

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        texts = []


        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extracting bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        centerX, centerY = centers[i][0], centers[i][1]
                        
                        if centerX <= W/3:
                            W_pos = "left "                        
                        elif centerX <= (W/3 * 2):
                            W_pos = "center "
                        else:
                            W_pos = "right "
                            
                        if centerY <= H/3:
                            H_pos = "top "
                        elif centerY <= (H/3 * 2):
                            H_pos = "mid "
                        else:
                            H_pos = "bottom "

                        texts.append(H_pos + W_pos + LABELS[classIDs[i]])
                        

                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                print(texts)
                cv2.imwrite("output/NewPicture.jpg",image)
                return texts

        # show the output image

