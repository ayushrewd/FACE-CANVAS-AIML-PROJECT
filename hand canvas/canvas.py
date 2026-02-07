import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480,640,3),dtype=np.uint8)

colors = [(255,0,255),(255,0,0),(0,255,0),(0,255,255)]
color_names = ["PURPLE","BLUE","GREEN","YELLOW"]
draw_color = colors[0]

xp, yp = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    # Top color bar
    for i,c in enumerate(colors):
        cv2.rectangle(frame,(i*160,0),(i*160+160,40),c,-1)
        cv2.putText(frame,color_names[i],(i*160+40,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        # Color select
        if y < 40:
            idx = x // 160
            draw_color = colors[min(idx,3)]

        # Draw
        if xp==0 and yp==0:
            xp,yp = x,y

        cv2.line(canvas,(xp,yp),(x,y),draw_color,8)
        xp,yp = x,y
    else:
        xp,yp = 0,0

    frame = cv2.add(frame,canvas)

    cv2.imshow("Air Canvas",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
