import torch

# =====================================================================
# 1. FÁZIS: ADATOK ELŐKÉSZÍTÉSE
# =====================================================================

adatok = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.5],
    [1.0, 1.0, 1.0]
]) # tanító adatok (bemenet1, bemenet2, cél)


w1 = torch.tensor(0.5, requires_grad=True) # súly az 1. pixelhez
w2 = torch.tensor(0.5, requires_grad=True) # súly a 2. pixelhez
b = torch.tensor(0.1, requires_grad=True) # közös torzítás/eltolás (bias)
t = 15 # tanulási ráta (learning rate) --- legjobb eredmény 16-tal
e = torch.exp

# =====================================================================
# 2. FÁZIS: AZ AI TAMÍTÁSA
# =====================================================================

for i in range(100000):

    x1, x2, y = adatok[torch.randint(0, adatok.size(0), (1,)).item()]


    # --- FORWARD PASS (előre terjesztés) ---
    z = x1 * w1 + x2 * w2 + b

    y2 = 1 / (1 + e(-z))      # ai tippje / sigmoid függvény
    
    L = 1/2*(y2 - y)**2       # költség függvény / kiszámoljuk a veszteséget (loss)

    """if L < 0.0000005:
        t = 50
    elif L < 0.000001:
        t = 50
    elif L < 0.000002:
        t = 50"""

    if i%1000==0: print(f"Kör {i:5d} | Bemenet: [{x1}, {x2}] | AI Tippje: {y2:.6f} | Hiba: {L:.8f}")


    # --- BACKWARD PASS (visszaterjesztés) - deriválás a láncszabállyal ---
    L.backward() # A lánc szabálya szerint a veszteség deriváltja a w1, w2 és b szerint (gradiens):

    with torch.no_grad():  # gradiens számítás nélkül
        # --- OPTIMIZATION (súlyok frissítése a tanulási ráta segítségével) ---
        w1 -= t * w1.grad
        w2 -= t * w2.grad
        b -= t * b.grad

        # --- GRADIENT ZEROING (gradiens nullázása a következő iterációhoz) ---
        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()


print("\n--- TANULÁS UTÁNI VÉGEREDMÉNY ---")
print(f"Végső súly (w1): {w1:.4f} | Végső súly (w2): {w2:.4f} | Végső torzítás (b): {b:.4f}")

# =====================================================================
# 2. FÁZIS: AZ AI HASZNÁLATA (Inference)
# =====================================================================

print("\nÜdvözöllek az AI kiértékelő zónájában!")
print("Adj meg két bemenetet vesszővel elválasztva.")
print("(A kilépéshez írd be, hogy: exit)\n")

while True:
    user_input = input("Írj be két számot (0 és/vagy 1), vesszővel elválasztva (x,y): ")

    if user_input.lower() == "exit":
        print("AI leállítva.")
        break
    try:
        input_x1, input_x2 = map(float, user_input.split(","))
        if input_x1 < 0 or input_x1 > 1 or input_x2 < 0 or input_x2 > 1:
            print(f"Kérlek, csak 0 és/vagy 1 értékeket adj meg!\n")
            continue

        for x1_adat, x2_adat, y_adat in adatok:
            if x1_adat == input_x1 and x2_adat == input_x2:
                elvart_y = y_adat
                break

        with torch.no_grad():
            z = input_x1 * w1 + input_x2 * w2 + b
            tipp = 1 / (1 + e(-z))  # AI tipp

        hiba = abs(tipp - elvart_y)
        esely = (1 - hiba) * 100
        vegso_tipp = round(2*tipp.item())/2

        print(f"-> AI belső összege (z): {z:.4f} | AI tippje: {2*tipp:.4f}")
        print(f"-> AI végső tippje (valószínűség): {2*vegso_tipp:.4f} ({esely:.2f}%)\n")

    except ValueError:
        print(f"Érvénytelen bemenet! Kérlek, csak csak 0 és/vagy 1 értékeket adj meg (x,y) formátumban!\n")
