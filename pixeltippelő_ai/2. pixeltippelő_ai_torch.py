import torch

# =====================================================================
# 1. FÁZIS: ADATOK ELŐKÉSZÍTÉSE
# =====================================================================

adatok = torch.tensor([[1.0, 1.0], [0.0, 0.0]]) # tanító adatok (bemenet, cél)


w = torch.tensor(0.5, requires_grad=True) # súly (weight) a pixelhez
b = torch.tensor(0.1, requires_grad=True) # torzítás/eltolás (bias) a pixelhez
t = 50 # tanulási ráta (learning rate)
e = torch.exp

# =====================================================================
# 2. FÁZIS: AZ AI TAMÍTÁSA
# =====================================================================

for i in range(100000):

    x, y = adatok[torch.randint(0, adatok.size(0), (1,)).item()] # véletlenszerűen kiválasztunk egy bemenetet és a hozzá tartozó célt


    # --- FORWARD PASS (előre terjesztés) ---
    z = x * w + b

    y2 = 1 / (1 + e(-z)) # ai tippje / sigmoid függvény
    
    L = 1/2*(y2 - y)**2          # költség függvény / kiszámoljuk a veszteséget (loss)

    if i%1000==0: print(f"Kör {i:3d} | AI Tippje: {y2:.6f} | Hiba (Loss): {L:.8f}")


    # --- BACKWARD PASS (visszaterjesztés) - deriválás a láncszabállyal ---
    L.backward() # automatikus deriválás a láncszabállyal

    with torch.no_grad():  # gradiens számítás nélkül
        # --- OPTIMIZATION (súlyok frissítése a tanulási ráta segítségével) ---
        w -= t * w.grad
        b -= t * b.grad

        # --- GRADIENT ZEROING (gradiens nullázása a következő iterációhoz) ---
        w.grad.zero_()
        b.grad.zero_()

print("\n--- TANULÁS UTÁNI VÉGEREDMÉNY ---")
print(f"Végső súly (w): {w:.4f} | Végső torzítás (b): {b:.4f}")

# =====================================================================
# 3. FÁZIS: AZ AI HASZNÁLATA (Inference)
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
            print(f"Kérlek, csak 0 vagy 1 értéket adj meg!\n")
            continue

        with torch.no_grad():
            z = input_x * w + b
            tipp = 1 / (1 + e(-z))  # AI tipp

        print(f"-> AI belső összege (z): {z:.4f} | AI tippje: {tipp:.4f}")
        
        if tipp < 0.5:
            vegso_tipp = 0
            esely = (1 - tipp) * 100
        else:
            vegso_tipp = 1
            esely = tipp * 100

        print(f"-> AI végső tippje (valószínűség): {vegso_tipp:.4f} ({esely:.2f}%)\n")

    except ValueError:
        print(f"Érvénytelen bemenet! Kérlek, csak számot adj meg (0 vagy 1)\n")