import math
import random

# =====================================================================
# 1. FÁZIS: ADATOK ELŐKÉSZÍTÉSE
# =====================================================================

adatok = [(0.0, 0.0, 0.0),
          (1.0, 0.0, 0.0),
          (0.0, 1.0, 0.0),
          (1.0, 1.0, 1.0)] # tanító adatok (bemenet1, bemenet2, cél)


w1 = 0.5 # súly az 1. pixelhez
w2 = 0.5 # súly a 2. pixelhez
b = 0.1 # közös torzítás/eltolás (bias)
t = 50 # tanulási ráta (learning rate)
e = math.exp

# =====================================================================
# 2. FÁZIS: AZ AI TAMÍTÁSA
# =====================================================================

for i in range(100000):

    x1, x2, y = random.choice(adatok)

    # --- FORWARD PASS (előre terjesztés) ---
    z = x1 * w1 + x2 * w2 + b

    y2 = 1 / (1 + e(-z)) # ai tippje / sigmoid függvény
    
    #L = - (y * math.log(y2) + (1 - y) * math.log(1 - y2))
    L = 1/2*(y2 - y)**2          # költség függvény / kiszámoljuk a veszteséget (loss)

    """if L < 0.0000005:
        t = 50
    elif L < 0.000001:
        t = 50
    elif L < 0.000002:
        t = 50"""

    if i%1000==0: print(f"Kör {i:5d} | Bemenet: [{x1}, {x2}] | AI Tippje: {y2:.6f} | Hiba: {L:.8f}")

    # --- BACKWARD PASS (visszaterjesztés) - deriválás a láncszabállyal ---
    L_y2 = 1*(y2 - y)            # a veszteség deriváltja az y2 szerint
    y2_z = y2 * (1 - y2)         # a sigmoid függvény deriváltja a z szerint

    z_w1 = x1                    # a z deriváltja a w1 szerint
    z_w2 = x2                    # a z deriváltja a w2 szerint
    z_b = 1                      # a z deriváltja a b szerint

    # A lánc szabálya szerint a veszteség deriváltja a w1, w2 és b szerint:
    L_w1 = L_y2 * y2_z * z_w1
    L_w2 = L_y2 * y2_z * z_w2
    L_b = L_y2 * y2_z * z_b

    # --- OPTIMIZATION (súlyok frissítése a tanulási ráta segítségével) ---
    w1 = w1 - t * L_w1
    w2 = w2 - t * L_w2
    b = b - t * L_b

print("\n--- TANULÁS UTÁNI VÉGEREDMÉNY ---")
print(f"Végső tipp 1000 kör után: {y2:.6f}")
print(f"Végső súly (w1): {w1:.4f} | Végső súly (w2): {w2:.4f} | Végső torzítás (b): {b:.4f}")

# =====================================================================
# 2. FÁZIS: AZ AI HASZNÁLATA (Inference)
# =====================================================================

print("\nÜdvözöllek az AI kiértékelő zónájában!")
print("Adj meg két bemenetet vesszővel elválasztva (pl. 1,0 vagy 1,1).")
print("(A kilépéshez írd be, hogy: exit)\n")

while True:
    user_input = input("Írj be két számot (0 és/vagy 1), vesszővel elválasztva: ")

    if user_input.lower() == "exit":
        print("AI leállítva.")
        break
    try:
        input_x1, input_x2 = map(float, user_input.split(","))
        if input_x1 < 0 or input_x1 > 1 or input_x2 < 0 or input_x2 > 1:
            print("Kérlek, csak 0 és/vagy 1 értékeket adj meg!")
            continue

        z = input_x1 * w1 + input_x2 * w2 + b
        tipp = 1 / (1 + e(-z))  # AI tipp 

        print(f"-> AI belső összege (z): {z:.4f}")
        print(f"-> AI végső tippje (valószínűség): {tipp:.4f} ({tipp*100:.2f}%)")

    except ValueError:
        print("Érvénytelen bemenet! Kérlek, csak számokat adj meg (0 és/vagy 1).")