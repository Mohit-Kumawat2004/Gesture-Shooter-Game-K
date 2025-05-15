import cv2
import numpy as np
import mediapipe as mp
import random
import time

# ===== Configurations Setup Occured=====
kacha_paka, uchaai = 1280, 720
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, kacha_paka)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, uchaai)

# MediaPipe Haath detection
mp_haath = mp.solutions.hands
haath = mp_haath.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# ===== Jugaadu Shooter Game Settings =====
bandook_ke_goli = 1
bacha_hua_goli = bandook_ke_goli
rang_goli = (0, 255, 255)  # Cyan bullets
rang_mun_na_hai = (0, 0, 255)  # Red if no ammo

barish_ke_goli = []  # [x, y, dx, dy]
nishano_ke_samuh = []
points_ka_meter = 0

# PowerUp settings
powerups = []
powerup_timer = {}
powerup_ka_effect = {
    'badi_goli': False,
    'bada_nishana': False,
    'double_point': False
}

# ===== RANG =====
rang_likhaai = (255, 255, 255)
rang_peechhe = (0, 0, 0)
rang_nishana = (255, 0, 255)  # Pink aim marker
rang_goli = (0, 255, 255)     # Cyan bullet

# ===== Helper Functions =====
def naya_nishana():
    x = random.randint(100, kacha_paka - 100)
    y = random.randint(100, uchaai - 100)
    return [x, y]

def dikhaye_interface(screen, ammo, points):
    cv2.rectangle(screen, (0, 0), (350, 120), (50, 50, 50), -1)
    multiplier = 2 if powerup_ka_effect['double_point'] else 1
    cv2.putText(screen, f"Goli: {ammo}/{bandook_ke_goli}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rang_likhaai, 2)
    cv2.putText(screen, f"Score: {points}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rang_likhaai, 2)
    if multiplier > 1:
        cv2.putText(screen, f"Multiplier: x{multiplier}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for i in range(bandook_ke_goli):
        rang = rang_likhaai if i < bacha_hua_goli else rang_mun_na_hai
        cv2.circle(screen, (150 + i * 30, 105), 10, rang, -1)

def pehchano_ishara(landmarks):
    ungli_8 = landmarks[8]
    angutha = landmarks[4]
    x1, y1 = int(ungli_8.x * kacha_paka), int(ungli_8.y * uchaai)
    x2, y2 = int(angutha.x * kacha_paka), int(angutha.y * uchaai)

    duri = np.hypot(x1 - x2, y1 - y2)
    gesture_tir = duri > 100

    duri_re = np.hypot(((landmarks[3].x - landmarks[5].x)kacha_paka)2, ((landmarks[3].y - landmarks[5].y)*uchaai)*2)
    gesture_re = duri_re < 1500
    return gesture_tir, gesture_re

def utaro_powerup():
    if len(powerups) < 3 and random.random() < 0.01:
        x = random.randint(50, kacha_paka - 50)
        ptype = random.choice(['badi_goli', 'bada_nishana', 'double_point'])
        powerups.append({'x': x, 'y': 0, 'type': ptype})

def dikhaye_powerups(screen):
    for p in powerups:
        rang = (255, 255, 0) if p['type'] == 'badi_goli' else (0, 255, 0) if p['type'] == 'bada_nishana' else (0, 0, 255)
        cv2.rectangle(screen, (p['x'], p['y']), (p['x']+30, p['y']+30), rang, -1)
        p['y'] += 5

# ===== Game Loop =====
while True:
    ok, frame = camera.read()
    if not ok:
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = haath.process(rgb_img)
    game_screen = frame.copy()

    dikhaye_interface(game_screen, bacha_hua_goli, points_ka_meter)

    if result.multi_hand_landmarks:
        for haath_points in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(game_screen, haath_points, mp_haath.HAND_CONNECTIONS)
            tir, reload = pehchano_ishara(haath_points.landmark)

            if reload:
                bacha_hua_goli = bandook_ke_goli
                cv2.putText(game_screen, "RELOAD!", (kacha_paka//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif tir and bacha_hua_goli > 0:
                bacha_hua_goli -= 1
                ungli = haath_points.landmark[8]
                ungli_base = haath_points.landmark[5]

                x = int(ungli.x * kacha_paka)
                y = int(ungli.y * uchaai)
                xb = int(ungli_base.x * kacha_paka)
                yb = int(ungli_base.y * uchaai)

                dx, dy = x - xb, y - yb
                norm = np.hypot(dx, dy)
                if norm:
                    dx /= norm
                    dy /= norm
                barish_ke_goli.append([x, y, dx * 15, dy * 15])

    # Tiro Update
    for g in barish_ke_goli:
        g[0] += int(g[2])
        g[1] += int(g[3])
        size = 10 if powerup_ka_effect['badi_goli'] else 5
        cv2.circle(game_screen, (int(g[0]), int(g[1])), size, rang_goli, -1)

    barish_ke_goli = [g for g in barish_ke_goli if 0 <= g[0] < kacha_paka and 0 <= g[1] < uchaai]

    if len(nishano_ke_samuh) < 3:
        nishano_ke_samuh.append(naya_nishana())

    naye_nishane = []
    for nishana in nishano_ke_samuh:
        mila = False
        for g in barish_ke_goli:
            dist = np.hypot(g[0] - nishana[0], g[1] - nishana[1])
            if dist < (50 if powerup_ka_effect['bada_nishana'] else 30):
                mila = True
                points_ka_meter += 2 if powerup_ka_effect['double_point'] else 1
                break
        if not mila:
            naye_nishane.append(nishana)
            cv2.circle(game_screen, tuple(nishana), 30, rang_nishana, -1)
            cv2.circle(game_screen, tuple(nishana), 20, (255, 255, 255), -1)
            cv2.circle(game_screen, tuple(nishana), 10, rang_nishana, -1)
    nishano_ke_samuh = naye_nishane

    utaro_powerup()
    dikhaye_powerups(game_screen)

    # Collision with powerups
    active_powers = []
    for p in powerups:
        if uchaai - 50 < p['y'] < uchaai:
            powerup_ka_effect[p['type']] = True
            powerup_timer[p['type']] = time.time()
        else:
            active_powers.append(p)
    powerups = active_powers

    # Timer check
    if powerup_ka_effect['badi_goli'] and time.time() - powerup_timer['badi_goli'] > 30:
        powerup_ka_effect['badi_goli'] = False
    if powerup_ka_effect['bada_nishana'] and time.time() - powerup_timer['bada_nishana'] > 20:
        powerup_ka_effect['bada_nishana'] = False
    if powerup_ka_effect['double_point'] and time.time() - powerup_timer['double_point'] > 20:
        powerup_ka_effect['double_point'] = False

    cv2.imshow("Desi Gesture Shooter", game_screen)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()