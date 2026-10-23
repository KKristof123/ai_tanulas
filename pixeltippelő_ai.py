import math
import random

# =====================================================================
# 1. FÁZIS: ADATOK ELŐKÉSZÍTÉSE
# =====================================================================

x = 1.0 # bemeneti/tanító pixel
y = 1.0 # cél pixel
adatok = [(1.0, 1.0), (0.0, 0.0)] # tanító adatok (bemenet, cél)


w = 0.5 # súly (weight) a pixelhez
b = 0.1 # torzítás/eltolás (bias) a pixelhez
t = 50 # tanulási ráta (learning rate)
e = math.exp

# =====================================================================
# 2. FÁZIS: AZ AI TAMÍTÁSA
# =====================================================================

for i in range(100000):

    x, y = random.choice(adatok)

    # --- FORWARD PASS (előre terjesztés) ---
    z = x * w + b

    y2 = 1 / (1 + e(-z)) # ai tippje / sigmoid függvény
    
    #L = - (y * math.log(y2) + (1 - y) * math.log(1 - y2))
    L = 1/2*(y2 - y)**2          # költség függvény / kiszámoljuk a veszteséget (loss)

    """if L < 0.0000005:
        t = 50
    elif L < 0.000001:
        t = 50
    elif L < 0.000002:
        t = 50"""
    
    if i%1000==0: print(f"Kör {i:3d} | AI Tippje: {y2:.6f} | Hiba (Loss): {L:.8f}")

    # --- BACKWARD PASS (visszaterjesztés) - deriválás a láncszabállyal ---
    L_y2 = 1*(y2 - y)            # a veszteség deriváltja az y2 szerint
    y2_z = y2 * (1 - y2)         # a sigmoid függvény deriváltja a z szerint
    z_w = x                      # a z deriváltja a w szerint
    z_b = 1                      # a z deriváltja a b szerint

    # A lánc szabálya szerint a veszteség deriváltja a w és b szerint:
    L_w = L_y2 * y2_z * z_w
    L_b = L_y2 * y2_z * z_b

    # --- OPTIMIZATION (súlyok frissítése a tanulási ráta segítségével) ---
    w = w - t * L_w
    b = b - t * L_b

print("\n--- TANULÁS UTÁNI VÉGEREDMÉNY ---")
print(f"Végső tipp 1000 kör után: {y2:.6f}")
print(f"Végső súly (w): {w:.4f} | Végső torzítás (b): {b:.4f}")

# =====================================================================
# 2. FÁZIS: AZ AI HASZNÁLATA (Inference)
# =====================================================================

print("\nÜdvözöllek az AI kiértékelő zónájában!")
print("Adj meg egy bemenetet (0 vagy 1), és az AI megmondja a tippet.")
print("(A kilépéshez írd be, hogy: exit)\n")

while True:
    user_input = input("Írj be egy számot (0 vagy 1): ")

    if user_input.lower() == "exit":
        print("AI leállítva.")
        break
    try:
        input_x = float(user_input)
        if input_x < 0 or input_x > 1:
            print("Kérlek, csak 0 vagy 1 értéket adj meg!")
            continue

        z = input_x * w + b
        tipp = 1 / (1 + e(-z))  # AI tipp 

        print(f"-> AI belső összege (z): {z:.4f}")
        print(f"-> AI végső tippje (valószínűség): {tipp:.4f} ({tipp*100:.2f}%)")

    except ValueError:
        print("Érvénytelen bemenet! Kérlek, csak számot adj meg (0 vagy 1).")