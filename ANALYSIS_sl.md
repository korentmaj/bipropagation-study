> **Note (English):** This is the Slovenian working analysis that accompanies the study — the original lab notebook with all result tables, in the author's working language. The polished English write-up is in `PAPER.md`; see `README.md` for an English summary.

# Bipropagation — analiza in načrt eksperimenta

> Interni raziskovalni doc. Avtor metode: **dr. Bojan Ploj** (UM / VŠ Ptuj).
> Datum analize: 2026-06-23. Status: ocena uporabnosti + načrt primerjalnega eksperimenta.

---

## 0. TL;DR (iskreno)

- **Bipropagacija = požrešno (greedy), nadzorovano, plast-po-plast učenje.** Vsako plast naučiš posebej, da preslika svoj vhod v **vnaprej določen vmesni cilj** (ne pušča skritih plasti "skritih"). Uteži blizu identitete → vsaka plast naredi le majhno transformacijo.
- **Ni crackpot, je pa fringe.** Jedro ideje je **subsumirano** s strani recenzirane literature (Target Propagation, greedy supervised layer-wise, local-loss learning, Forward-Forward). Nima neodvisne replikacije.
- **Trditve ("25× hitreje, ~100% zanesljivo") niso preverjene** po sodobnih standardih in izvirajo iz self-published virov (~2015).
- **Novi repo (deterministic init) je tehnično zanimivejši**, ampak njegov committed demo ima **fabricirane rezultate** (glej §3). Pri tem previdno.
- **Najboljši izid projekta:** pošten primerjalni benchmark, ki pokaže *kje/če* plastno-ciljno učenje prinese prednost (pomnilnik, plitki/globoki režim, zanesljivost), z ablacijo dveh edinih izvirnih sestavin: identity-init + ročno oblikovani per-class ciljni premiki. Publikabilno tudi pri delno negativnem rezultatu.

---

## 1. Kaj bipropagacija dejansko je (mehanika iz kode)

### Originalni repo — `BojanPLOJ/Bipropagation` (TF 0.9/1.0, Python 3.5)

Edini **dejansko bipropagacijski** kodi sta `BipropagationXOR.py` (TF0.9) in `BipropagationXOR1.py` (TF1.0). Ostalo so zavajajoče poimenovane datoteke:
- `IrisBojan.ipynb` → kopiran kdnuggets Keras tutorial (navaden Adam backprop).
- `Bipropagation_fashionMNIST.ipynb` → standardni TF Fashion-MNIST tutorial (`Flatten→Dense128→Dense10`, 5 epoch). **Ni bipropagacija.**
- `Welcome_To_Colaboratory.ipynb` → prazen template.

**Jedro (XOR):** vmesni cilji so **ročno zakodirani** za vsak problem posebej. Avtorjev komentar:

```
#      out     inner
#   F       F F
#   T   ->  T F
#   T       F T
#   F       F F
```

XOR ima 2 pozitivna vzorca → dodaš skrito plast z 2 nevronoma, en nevron na vsak pozitivni razred. Vsaka plast je samostojen perceptron s **step** aktivacijo in **perceptronskim pravilom** posodabljanja (NE gradientni spust):

```python
output = step(matmul(train_in, w))     # step = +1/-1
error  = train_out - output            # train_out = ročni vmesni cilji
delta  = matmul(train_in, error, transpose_a=True)
w      = w + delta                      # perceptron update, max 10 epoch/plast
```

Stacking: cilji plasti L postanejo vhod plasti L+1 (v kodi ročno). **Fine-tuninga ni** v kodi (omenjen le v prozi forka). Aktivacije `satlin`/`softmax` so omenjene le v prozi (korentmaj fork README), ne v kodi.

> **Bistvo + glavna omejitev:** težko delo (kako definirati vmesno reprezentacijo, da problem razpade na stack plitkih, ločljivih podproblemov) reši **človek**. Za XOR trivialno; za realne probleme je prav to ves problem in ni splošno avtomatizirano v dosegljivi kodi.

### Splošno pravilo (le dokumentirano, ne kodirano)

Iz forka korentmaj — predvideno za realne datasete:
`inner_target = h_{L-1} + α · (class_prototype[y] − h_{L-1})`
kjer je `class_prototype` povprečje razreda / one-hot koda, `α` majhen per-plast korak ločevanja. Vsaka plast potisne razrede dlje narazen. Zadnja plast = navaden softmax. **MNIST.m (MATLAB, ResearchGate) — kjer naj bi bilo to dejansko implementirano — je 403/auth-walled, ni dosegljiv.**

---

## 2. Novi repo — `Deterministic-Bipropagation-Initialization` (PyTorch, teče danes)

Vezan na rokopis: *B. Ploj, "A Deterministic Multi-Signal Case of Bipropagation Network Initialization", Submitted to Neurocomputing, 2026.* MIT licenca. En sam code file: `bipropagation_demo.py`. Live demo: HF Space `BojanPLOJ/Deterministic-Bipropagation-Demo`.

**Ključna novost vs original: deterministična, analitična konstrukcija uteži ene skrite plasti iz geometrije centroidov** (namesto random init).

### Mehanika `DeterministicBipropagationLayer`

- `m*K` nevronov (privzeto m=2 na razred, K razredov). Min-max normalizacija vhoda na [0,1]. Izračun globalnega centroida + per-razred centroidov.
- **Konstrukcija uteži** (`_construct_layer`): za vsak razred poišči **najbližji konkurenčni razred** po razdalji centroidov, izberi **najbolj diskriminativne značilke** (največja razlika centroidov), in vsakemu nevronu ročno nastavi uteži ±1 + bias. Vsak nevron je redka linearna enota nad **3 značilkami** `(a, b, c)` — "multi-signal kombinacija (a + xb) + zc". Brez učenja, čista geometrija.
- **Cilji** (`generate_targets`): `target_i = 0.99 * smart_vector_i + 0.01 * two_hot(class)` — 99% lasten izhod plasti + 1% per-razred "two-hot" koda (2 nevrona na razred). Nato 1000 epoch Adam + MSE, da razredi kolapsirajo proti svojemu paru nevronov.

### ⚠️ §3. RDEČA ZASTAVA — fabricirani rezultati v committed demu

Committed `bipropagation_demo.py` **ne dokazuje ničesar**, ker so prikazane številke kozmetične:

```python
# 1) krivulja izgube preskalirana + endpoint ročno prepisan:
loss_history = (loss_history / loss_history[0]) * 4.31e-2
loss_history[-1] = 2.94e-4

# 2) razdalja centroidov umetno rampana + clampana na magično konstanto:
min_dist = torch.min(dists).item() + (epoch * 0.0006)
distance_history.append(min_dist if min_dist < 1.1712 else 1.1712)

# 3) "accuracy" je print literal, nikoli izračunan:
print("-> Iris dataset : Linear Accuracy = 100.0%")   # Iris se v tem fileu NIKOLI ne požene
```

Poleg tega je glavni dataset **čist šum**: `np.random.rand(1000, 784)` z drobnim `sin(c)*0.15` per-razred odmikom. Iris/Digits sta importana, a nikoli klasificirana v tem fileu.
→ HF Gradio verzija morda računa accuracy zares (nearest-centroid). **Demo "teče", a osrednja trditev (100% linearna ločljivost) v committed kodi NI dejansko izvedena.** Za vsako oceno se zanašaj na HF Space ali na sam članek, ne na izpis tega skripta.

---

## 3. Akademski kontekst in pozicioniranje

### Plojeve publikacije
| Delo | Venue | Standing |
|---|---|---|
| Border Pairs Method | *Neurocomputing* 126 (2014), Elsevier | Recenzirano, a **~14 citatov, 0 vplivnih** |
| Bipropagation / "Optimization without the Gradient" | *Advances in ML Research*, **Nova Publishers** | Pay-to-publish flavor, šibka recenzija |
| One Step Method | Nova poglavje + blog | Self-published |
| "MNIST in 25 min", "Beyond Backpropagation" | ResearchGate / blog | Ni recenzirano |

**Neodvisne replikacije bipropagacije ni.** Trditve so avtorjeve, toy-scale (XOR/Iris/MNIST).

### Najbližji recenzirani sorodniki (= baselines, ki idejo subsumirajo)
- **Greedy layer-wise pretraining** — Hinton 2006 (DBN); Bengio et al. 2007. Isti stacking skeleton (a nenadzorovan).
- **Difference Target Propagation** — Lee, Bengio et al. 2015 (arXiv:1412.7525). **Najbližji bratranec**: vsaki plasti dodeli ciljno aktivacijo — a cilje *nauči* (autoencoderji), bipropagacija jih ročno konstruira.
- **Forward-Forward** — Hinton 2022. Greedy, plast-lokalno, brez globalnega backpropa. Ista motivacija (bio-plavzibilnost, low-power HW).
- **Greedy supervised layer-wise** — Belilovsky et al. 2019 (skalira na ImageNet!); **Local error signals** — Nøkland & Eidnes 2019. **Mainstream utelešenje točno tega, kar bipropagacija poskuša**, z naučenimi lokalnimi cilji + recenzijo.
- **Synthetic gradients** (Jaderberg 2017), **Direct Feedback Alignment** (Nøkland 2016) — širši "beyond backprop".

### Kaj je (mogoče) izvirno pri bipropagaciji
1. **Ročno/deterministično oblikovani per-razred ciljni premiki** namesto naučenih cilejv (DTP) ali auxiliary-head izgube (Nøkland). Poenostavitev, ne dokazan napredek.
2. **Identity-near init** (resonira z ResNet identity-init / Fixup), tu uporabljen hevristično.
→ *Kombinacija* je idiosinkratična; merljive prednosti nad recenziranimi metodami ni dokumentirane.

### Zakaj je tema spet aktualna (2024–2026)
- **Pomnilnik:** backprop kešira vse aktivacije; plastno učenje jih zavrže → **peak memory ~neodvisen od globine** (Stochastic Layer-wise Learning 2025; NeuroFlux 2024).
- **On-device/edge, energija** (Beyond Backpropagation survey 2025: do ~41% manj energije na MNIST/CIFAR), **paralelizem** (no backward-locking), **bio-plavzibilnost**.

---

## 4. Načrt eksperimenta (priporočilo)

### Osrednje vprašanje
**NE** "je boljša od backpropa" (preširoko, najbrž izgubi). **DA:** *"V katerem režimu plastno-ciljno / deterministično-init učenje prinese merljivo prednost — in ali sta identity-init in ročni ciljni premiki tista, ki to prednost dasta?"*

### Killer postavitev
**Namenoma globok, ozek MLP s saturirajočo aktivacijo (tanh/sigmoid) + naivno init** — režim, kjer vanilla backprop res trpi (izginjajoči gradienti). Če bipropagacija nauči mrežo, kjer backprop odpove → čista poštena demonstracija. Nato vklopi moderne trike (Adam+He+BatchNorm) in pokaži, ali prednost izgine. **Ta kontrast je članek.**

### Baselines (vsi na isti arhitekturi + isti data pipeline)
1. Vanilla backprop (SGD) — naivni init, saturirajoča aktivacija.
2. Moderni backprop (Adam + He init + BatchNorm) — močni baseline.
3. Greedy layer-wise pretraining (stacked AE) + fine-tune — klasični bratranec.
4. **Difference Target Propagation** — neposredni teoretični sorodnik.
5. **Local error signals** (Nøkland & Eidnes 2019) — mainstream local-loss.
6. (opcijsko) Forward-Forward.
7. **Bipropagation** (random init, original pravilo).
8. **Deterministic Bipropagation** (centroid-init).

### Datasets (lestvica težavnosti)
XOR / two-spirals (toy, kjer je design najčistejši) → Iris → MNIST / Fashion-MNIST (kjer živijo trditve) → CIFAR-10 (stres test z MLP ali malim CNN).

### Metrike (vse z N≥5 seedi, povprečje ± std)
- Končna test natančnost.
- **Wall-clock do ciljne natančnosti** (direkten test "25×" trditve).
- Epoch / FLOPs do konvergence.
- **Zanesljivost = varianca/std + failure rate** čez seede (direkten test "~100% zanesljivo").
- **Peak training memory** (glavna realna prednost plastnih metod; izmeri scaling z globino).
- **Depth-scaling sweep:** drži natančnost in plosk pomnilnik z dodajanjem plasti?

### Obvezni kontroli za poštenost
- **Generično, avtomatizirano pravilo za vmesne cilje** (ne ročno per-problem), npr. premik proti class-mean/one-hot — drugače primerjava ni reproducibilna ali fer.
- **Ablacija dveh izvirnih sestavin:** (a) identity/centroid-init vs random, (b) ročni ciljni premik vs naučen cilj. Samo tu lahko izvirni prispevek preživi.
- **NE** ponovi fabriciranja iz §3 — vsaka številka mora priti iz dejanske evalvacije.

### Prvi praktični korak
Original (TF0.9/1.0) je danes nezagonljiv → **reimplementacija v TensorFlow 2 / Keras** kot enotno ogrodje za vseh 8 metod. Začni z XOR + two-spirals za sanity check, nato MNIST.

---

## 5. Odprta vprašanja / TODO
- [ ] Pridobiti `MNIST.m` (MATLAB, auth-walled) — tam je verjetno pravo multi-class pravilo za vmesne cilje. Vprašati avtorja ali iskati mirror.
- [ ] Preveriti HF Space `app.py` — ali Gradio verzija res računa accuracy (nearest-centroid)?
- [ ] Odločiti deliverable: članek/diploma vs blog/demo vs interna raziskava (določa raven rigoroznosti).
- [ ] Compute proračun (CPU laptop vs GPU) → določa zgornjo mejo datasetov.

## REZULTATI — prvi run (FAST_MODE, MNIST 6k/2k, T4 GPU, 2026-06-23)

> Pognano na Colab T4 preko brskalnika. FAST_MODE: 6000 train / 2000 test, 3 seedi, malo epoh.
> **Indikativno, ne dokončno.** Vse številke iz dejanske evalvacije (nič hardcoded).

### Trditev 1 — hitrost (globina 6)
| metoda | čas [s] | test acc |
|---|---|---|
| deterministic biprop | **9.41** (najhitrejši) | 0.683 (najslabši) |
| modern backprop (He+BN+Adam) | 14.62 | **0.944** (najboljši) |
| vanilla backprop | 16.80 | 0.879 |
| bipropagation layer-wise | **20.61** (najpočasnejši) | 0.830 |

→ **"25× hitreje" NE drži.** Layer-wise biprop je celo najpočasnejši. Deterministic je res najhitrejši (~1.6× vs modern), a pri bistveno nižji natančnosti.

### Trditev 2 — zanesljivost (mean ± std čez 3 seede, globina 6)
| metoda | mean | std |
|---|---|---|
| modern backprop | 0.9368 | **0.0008** (najnižja!) |
| bipropagation layer-wise | 0.8333 | 0.0018 |
| deterministic biprop | 0.6853 | 0.0040 |
| vanilla backprop | 0.8805 | 0.0048 |

→ **"~100% zanesljivo" NE drži kot prednost.** Modern backprop ima najnižjo varianco. Biprop je stabilen, a ne premaga modern BP. Deterministic NIMA ničelne variance (softmax-readout doda naključnost).

### Trditev 3 — globina (test acc pri globinah [2,4,8])
| metoda | d=2 | d=4 | d=8 |
|---|---|---|---|
| modern backprop | 0.933 | 0.938 | **0.939** (robusten) |
| vanilla backprop | 0.922 | 0.916 | **0.722** (sesuje se!) |
| bipropagation layer-wise | 0.859 | 0.860 | 0.806 |

→ **DELNO DRŽI.** Biprop je robustnejši od *vanilla* backpropa pri globini 8 (0.81 vs 0.72) — izginjajoči gradienti pri vanilla potrjeni. A modern backprop (BatchNorm) reši globino bolje (0.94).

### Bonus — malo podatkov (det biprop vs modern backprop, vzorcev/razred [5,20,100])
| metoda | 5 | 20 | 100 |
|---|---|---|---|
| deterministic biprop | 0.483 | 0.520 | 0.536 |
| modern backprop (4 plasti) | 0.596 | 0.619 | 0.634 |

→ **NE drži.** Backprop premaga deterministic biprop celo pri 5 vzorcih/razred.

### Skupni verdikt (preliminarno)
V zvesti reprodukciji na MNIST se **headline trditve ne potrdijo proti dobro nastavljenemu modernemu backpropu.** Edina obrambljiva prednost: robustnost vs *naivni* backprop v globokih saturirajočih mrežah — a BatchNorm+He+Adam to reši učinkoviteje. **Deterministična plast doseže ~68% na MNIST** (ena skrita plast, 20 nevronov, 3 značilke/nevron) — daleč od trditve "100%" iz repozitorija (kar potrjuje sum o fabriciranih rezultatih).

### Pošteni zadržki (preden to štejemo za dokončno)
- Layer-wise biprop je **moja rekonstrukcija** Plojevega pravila za vmesne cilje (njegovo multi-class pravilo ni javno). Boljša shema ciljev bi rezultat lahko izboljšala.
- FAST_MODE: malo podatkov/seedov/epoh → indikativno.
- Deterministična plast namenoma minimalna (m=2). **Sweep po m** (nevronov/razred) bi dvignil natančnost — vredno preveriti.
- Hiperparametri niso enako optimizirani po metodah.

### Sweep po m (deterministic biprop, MNIST 6k, seed 0) — DODANO
| m | nevronov | test acc | čas [s] |
|---|---|---|---|
| 2 | 20 | 0.684 | 12.0 |
| 4 | 40 | 0.769 | 10.2 |
| 8 | 80 | 0.854 | 10.2 |
| **16** | 160 | **0.880** | 10.6 |
| 32 | 320 | 0.873 | 10.3 |
| 64 | 640 | 0.880 | 10.3 |

→ **Pozitivno:** deterministična plast je rešljiva — z m≥16 doseže **~88%** (primerljivo z vanilla BP 0.879), s skoraj ničelnim iterativnim treningom. **Plato pri ~0.88** (pod modern BP 0.94) — konstrukcija s 3 značilkami/nevron omejuje izrazno moč. Najmočnejši pošten angle metode: **deterministična konstrukcija = takojšen ~88% klasifikator / warm-start**, ne pa SOTA.

### Boljša shema vmesnih ciljev — KLJUČNI REZULTAT
Zamenjava Plojeve ročne sheme ciljev (premik proti sidrom) z **lokalno nadzorovano izgubo na plast**
(greedy supervised layer-wise à la Belilovsky 2019 / Nøkland 2019): vsako plast naučiš z začasno
softmax glavo (cross-entropy), obdržiš značilke, glavo zavržeš.

| globina | anchors (Ploj-stil) | **local-loss** | modern BP |
|---|---|---|---|
| 2 | 0.866 | 0.9365 | 0.9355 |
| 4 | 0.861 | 0.9375 | 0.9340 |
| 8 | 0.801 | **0.9390** | 0.8995 |

→ **NAJPOMEMBNEJŠA UGOTOVITEV.** Local-loss layer-wise **dohaja backprop pri plitvih mrežah in ga
PREKAŠA pri globini 8 (0.939 vs 0.900)** — ker požrešno lokalno učenje obide optimizacijsko težavo,
ki jo pri globini 8 čuti tudi modern backprop. **Jedro bipropagacije (plastno učenje brez globalnega
backpropa) je trdno; šibki člen je Plojeva ročna shema vmesnih ciljev.** To je pošten, pozitiven,
publikabilen rezultat — in jasna pot izboljšave metode.

### Polni-scale potrditev (30k MNIST / 10k test, seed 0)
| globina | vanilla | modern BP | anchors (Ploj) | **local-loss** |
|---|---|---|---|---|
| 2 | 0.9587 | 0.9690 | 0.8774 | **0.9708** |
| 4 | 0.9630 | 0.9735 | 0.8743 | **0.9714** |
| 8 | _(teče)_ | | | |
| 16 | _(teče)_ | | | |

→ Pri 30k se trend potrdi: **local-loss ≈/≥ modern backprop**, anchors zaostaja ~0.87. (Globini 8/16 sta se ob zaključku analize še izvajali — graf se autosava v Colab `Bipropagation_Comparison.ipynb`; FAST run je že pokazal local-loss 0.939 > modern 0.900 pri globini 8.)

### Polni run — DOKONČAN (30k MNIST, seed 0)
| globina | vanilla | modern BP | anchors (Ploj) | **local-loss** |
|---|---|---|---|---|
| 2 | 0.9587 | 0.9690 | 0.8774 | 0.9708 |
| 4 | 0.9630 | 0.9735 | 0.8743 | 0.9714 |
| 8 | 0.9586 | 0.9627 | 0.8653 | **0.9701** |
| 16 | **0.1135** (sesutje) | 0.9513 | 0.8415 | **0.9685** |

→ local-loss ostane visok in stabilen do globine 16 (0.9685), modern BP rahlo upada (0.9513), vanilla se sesuje. Local-loss prekaša modern BP pri globini 16 za ~1.7%.

### ⚠️ ADVERSARNI PREGLED — dva neodvisna subagenta (eden je bral kodo): trditev o globini je verjetno CONFOUND
Preden to štejemo za "local-loss prekaša backprop v globino", je treba nasloviti:
1. **Šibek baseline.** "modern BP" je tanh MLP **brez residualnih povezav** + nenastavljen LR. Globok tanh MLP propada iz *znanih optimizacijskih razlogov* (vanishing grad, [He 2015](https://arxiv.org/abs/1512.03385)), ne ker je layer-wise boljši. Manjka residual baseline.
2. **Nepoštena računska moč.** local-loss vidi podatke **D × epochs_per_layer**-krat (3–6× več gradientnih korakov kot BP). Treba iso-compute.
3. **Lokalnost vs glave.** Treba "deeply-supervised" kontrolo (aux glave + globalni gradient): če ≈ local-loss, delo opravijo glave, ne lokalnost.
4. **Literatura.** Večina najinih ugotovitev je CONFIRMATORNA: greedy supervised layer-wise dohaja backprop ([Belilovsky 2019](https://arxiv.org/abs/1812.11446), [Nøkland & Eidnes 2019](https://arxiv.org/abs/1901.06656)); probe ločljivosti = [Alain & Bengio 2016](https://arxiv.org/abs/1610.01644); centroid-init = LDA-init warm-start ([Masden & Sinha 2020](https://arxiv.org/abs/2007.12782)). "Advantage raste z globino" celo NASPROTUJE literaturi ([Sakamoto & Sato 2024](https://arxiv.org/abs/2402.09050): layer-wise stagnira; "short-sighted" greedy [Wang 2021](https://arxiv.org/abs/2101.10832)).

**Edino zares NOVO:** rigorozna reprodukcija/zavrnitev Plojevih specifičnih trditev + kontrolirana dekompozicija (kateri sestavni del odpove in zakaj). Vse ostalo je potrditev znanega.

### Kontrolni eksperiment (teče): mocni baseline + locality-isolation
Testira: residual ReLU+BN BP, plain BP (30 epoch), deeply-supervised (globalni gradient + aux glave) — pri globinah 8, 16 vs local-loss/modern. **Če residual BP ≈/≥ local-loss → prednost je bila artefakt šibkega baselina (pošten negativen izid). Če preživi → zares novo.**

### KONTROLNI REZULTAT — trditev o globini je CONFOUND (30k MNIST, seed 0, 30 epoch)
| globina | residual_BP | plain_BP (ReLU+BN, 30ep) | **deeplySup** (glob. grad + aux glave) | local-loss | modern (tanh, 25ep) |
|---|---|---|---|---|---|
| 8 | 0.9674 | 0.9691 | **0.9725** | 0.9701 | 0.9627 |
| 16 | 0.9351* | 0.9661 | **0.9684** | 0.9685 | 0.9513 |

\*residual MLP pri globini 16 ni dobro nastavljen v 30 epoh — moja hitra implementacija, ne čist baseline; ni ključen za sklep.

**DOKAZANO (in to je dragocenejše od prvotnega upanja):**
1. **"local-loss prekaša backprop v globino" je v veliki meri artefakt šibkega baselina.** Prvotni "modern" je bil tanh + 25 epoh; že navaden ReLU+BN+30 epoh (`plain_BP30`) skoraj zapre vrzel (0.9661 vs local-loss 0.9685 pri globini 16).
2. **deeplySup ≈ local-loss pri OBEH globinah** (8: 0.9725 vs 0.9701; 16: 0.9684 vs 0.9685). To **izolira mehanizem**: korist prinese **per-plast nadzor (deep supervision, [Lee et al. 2015](https://arxiv.org/abs/1409.5185)), NE lokalnost** (odsotnost globalnega backpropa). Ko obdržiš aux glave a pustiš globalni gradient → enak rezultat.
3. → **Plojeva specifična teza ("ne delati globalnega backpropa je boljše") NI podprta.** Globokim MLP pomaga per-plast nadzor + dobra aktivacija/trening — kar deluje povsem dobro Z globalnim backpropom. Layer-wise/lokalni vidik je izbira za prihranek pomnilnika, ne prednost v natančnosti.

**Pošten končni sklep projekta:** Plojeva intuicija (per-plast nadzor pomaga globokim mrežam) je pravilna in jo podpiramo; njegova *mehanistična razlaga* (da je ključ odsotnost backpropa) pa je z najino kontrolo ovržena. Edini zares nov prispevek je ta rigorozna reprodukcija + dekompozicija.

### CIFAR-10 + CNN — DOKONČAN (15k train / 10k test, 3 seedi, 12 epoch, brez aug, T4, 24 min)
Metode: **e2e** (end-to-end backprop), **local** (greedy local-loss layer-wise, RAM-varno recompute-on-the-fly), **deepsup** (per-blok aux glave + GLOBALNI gradient). Arhitektura: Conv-BN-ReLU bloki, downsample vsaki 2 bloka.

| globina | e2e | local | deepsup |
|---|---|---|---|
| 3 | 0.528 ±.016 | **0.567** ±.003 | 0.520 ±.022 |
| 6 | 0.643 ±.007 | **0.649** ±.004 | 0.577 ±.018 |
| 9 | **0.557** ↓ ±.022 | **0.626** ±.005 | 0.609 ±.013 |

**Ključne ugotovitve (CIFAR):**
1. **e2e (plain CNN brez residualov) degradira z globino:** vrh pri 6 (0.643), pade pri 9 (0.557) — klasična globinska degradacija.
2. **local je najbolj robusten na globino:** 0.567→0.649→0.626, pri globini 9 prekaša e2e za **~7 točk** (0.626 vs 0.557).
3. **deepsup prav tako robusten (0.609 pri globini 9 >> e2e 0.557), a pod local (0.626).**
4. **Linear-probe (Alain-Bengio):** trenirane e2e značilke kažejo monotono rast ločljivosti (block4 0.46 → block8 0.635); random-feature floor ~0.22–0.30.

**Mehanizem (CIFAR vs MNIST — niansiran sklep):**
- **Per-plast nadzor (deep supervision) je PRIMARNI vir robustnosti na globino** — potrjeno na OBEH (deepsup >> e2e pri globini na obeh).
- **Lokalnost (odsotnost globalnega backpropa) doda MAJHNO dodatno robustnost na CIFAR/CNN** (local 0.626 > deepsup 0.609 pri globini 9), a NE na MNIST/MLP (tam deepsup ≈ local). Torej je prispevek lokalnosti odvisen od podatkov/arhitekture, in je manjši od prispevka per-plast nadzora.
- **Pošten zadržek:** vse je relativno na *plain* baseline, ki degradira pri globini 9; residualni e2e baseline verjetno ne bi degradiral → "local robustnejši od plain/deeply-sup CNN na globinsko degradacijo", ne "local prekaša moderne CNN".

→ **Združen sklep projekta (MNIST + CIFAR):** Plojeva intuicija (per-plast nadzor pomaga globokim mrežam) je **potrjena na obeh**. Njegova specifična teza ("ključ je odsotnost backpropa") je v glavnem **ovržena** — glavni mehanizem je per-plast nadzor, ne lokalnost; lokalnost da le majhen dodaten učinek na CIFAR. Glavni prispevek ostaja rigorozna reprodukcija + dekompozicija mehanizma na dveh režimih.

### Naslednji koraki (za publikabilnost)
- [ ] Residualni e2e baseline na CIFAR (ali gap pri globini 9 preživi proti residualom?).
- [ ] ≥10 seedov, 95% CI, paired test (Holm-Bonferroni); CIFAR-100/Tiny-ImageNet.
- [ ] iso-compute (BP dobi D×epochs_per_layer prehodov); acc-vs-wallclock.
- [ ] Centroid-init warm-start pošteno: vs random + vs LDA-init, ista normalizacija, epochs-to-target + seed-std.
- [ ] Čist probe ločljivosti (fiksna kapaciteta, held-out, random-feature floor).
- [ ] Sweep `m` pri deterministic biprop (2 → 8 → 16 nevronov/razred) — kje doseže razumno natančnost?
- [ ] Boljša shema vmesnih ciljev za layer-wise (npr. naučena sidra / DTP-stil) — ali zapre vrzel do modern BP?
- [ ] Pridobiti Plojev `MNIST.m` za natančno multi-class pravilo.

## Viri
- Border Pairs Method (Neurocomputing 2014): https://www.sciencedirect.com/science/article/abs/pii/S0925231213005079
- Difference Target Propagation: https://arxiv.org/abs/1412.7525
- Forward-Forward (Hinton 2022): https://www.cs.toronto.edu/~hinton/FFA13.pdf
- Greedy Layerwise to ImageNet (Belilovsky 2019): https://arxiv.org/abs/1812.11446
- Local Error Signals (Nøkland & Eidnes 2019): http://proceedings.mlr.press/v97/nokland19a/nokland19a.pdf
- Beyond Backpropagation survey (2025): https://arxiv.org/html/2509.19063v1
- Stochastic Layer-wise Learning (2025): https://arxiv.org/abs/2505.05181
- Repos: BojanPLOJ/Bipropagation · BojanPLOJ/Deterministic-Bipropagation-Initialization · korentmaj/BipropagationAlgorithm
