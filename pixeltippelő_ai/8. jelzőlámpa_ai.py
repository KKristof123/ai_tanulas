import torch

# =====================================================================
# 1. FÁZIS: ADATOK ELŐKÉSZÍTÉSE
# =====================================================================

adatok = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # Piros -> Megállás (0)
    [1.0, 1.0, 0.0, 0.33],  # Piros-Sárga -> Felkészülés (0.33)
    [0.0, 1.0, 0.0, 0.66],  # Sárga -> Lassítás (0.66)
    [0.0, 0.0, 1.0, 1.0]    # Zöld -> Mehet (1)
])

w1 = torch.tensor(0.5, requires_grad=True) # Súly a Piros lámpához
w2 = torch.tensor(0.5, requires_grad=True) # Súly a Sárga lámpához
w3 = torch.tensor(0.5, requires_grad=True) # Súly a Zöld lámpához
b = torch.tensor(0.1, requires_grad=True)  # Közös torzítás (bias)

t = 15 # tanulási ráta (learning rate) --- legjobb eredmény 16-tal
e = torch.exp

# =====================================================================
# 2. FÁZIS: AZ AI TAMÍTÁSA
# =====================================================================

for i in range(100000):

    piros, sarga, zold, y = adatok[torch.randint(0, adatok.size(0), (1,)).item()]


    # --- FORWARD PASS (előre terjesztés) ---
    z = piros * w1 + sarga * w2 + zold * w3 + b

    y2 = 1 / (1 + e(-z))      # ai tippje / sigmoid függvény
    
    L = 1/2*(y2 - y)**2       # költség függvény / kiszámoljuk a veszteséget (loss)

    if i%1000==0: print(f"Kör {i:5d} | Bemenet: [{piros}, {sarga}, {zold}] | AI Tippje: {y2:.6f} | Hiba: {L:.8f}")


    # --- BACKWARD PASS (visszaterjesztés) - deriválás a láncszabállyal ---
    L.backward() # A lánc szabálya szerint a veszteség deriváltja a w1, w2 és b szerint (gradiens):

    with torch.no_grad():  # gradiens számítás nélkül
        # --- OPTIMIZATION (súlyok frissítése a tanulási ráta segítségével) ---
        w1 -= t * w1.grad
        w2 -= t * w2.grad
        w3 -= t * w3.grad
        b -= t * b.grad

        # --- GRADIENT ZEROING (gradiens nullázása a következő iterációhoz) ---
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        b.grad.zero_()


print("\n--- TANULÁS UTÁNI VÉGEREDMÉNY ---")
print(f"Végső súlyok -> Piros(w1): {w1.item():.4f} | Sárga(w2): {w2.item():.4f} | Zöld(w3): {w3.item():.4f} | Bias(b): {b.item():.4f}")

# =====================================================================
# 2. FÁZIS: AZ AI HASZNÁLATA (Inference)
# =====================================================================

print("\nÜdvözöllek a Jelzőlámpa AI kiértékelő zónájában!")
print("Adj meg 3 lámpa állapotot (0 vagy 1) vesszővel elválasztva (Piros,Sárga,Zöld).")
print("(A kilépéshez írd be, hogy: exit)\n")

while True:
    user_input = input("Add meg a lámpák állapotát (Piros,Sárga,Zöld): ")

    if user_input.lower() == "exit":
        print("AI leállítva.")
        break
    try:
        piros, sarga, zold = map(float, user_input.split(","))

        # Ellenőrizzük, hogy tényleges lámpa kombinációt adtak-e meg
        elvart_y = None
        for sor in adatok:
            if sor[0].item() == piros and sor[1].item() == sarga and sor[2].item() == zold:
                elvart_y = sor[3].item()  # Ha megvan, elmentjük a hozzá tartozó cél y-t
                break

        if elvart_y is None:
            print(f"Ez a kombináció nem a 4 legális lámpa-szituáció egyike! Szabálytalan lámpa!\n")
            continue  # Visszaugrik a ciklus elejére, új inputot kérni

        with torch.no_grad():
            z = piros * w1 + sarga * w2 + zold * w3 + b
            tipp = 1 / (1 + e(-z))  # AI tipp

        hiba = abs(tipp - elvart_y)
        pontossag = (1 - hiba) * 100

        print(f"-> AI belső összege (z): {z:.4f}")
        print(f"-> AI tippje (pontosság): {tipp:.4f} ({pontossag:.2f}%)\n")

    except ValueError:
        print(f"Érvénytelen bemenet! Kérlek 3 számot adj meg vesszővel elválasztva (pl: 1,1,0)!\n")
