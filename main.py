from ultralytics import YOLO
import cv2
model=YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    print("ret =", ret)
    if not ret:
        break
    # YOLO attend des images BGR (comme OpenCV)
    results = model(frame)  # retourne les détections
    # Dessiner les boîtes sur l'image
    annotated_frame = results[0].plot()  # plot() retourne l'image annotée
    # Afficher l'image
    cv2.imshow("YOLO Webcam", annotated_frame)
    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()