**Zadeva:** Neodvisna reprodukcija vaše bipropagacije — pozitivne ugotovitve in nekaj vprašanj, ki bi jih rada predebatirala z vami

Spoštovani dr. Ploj,

sem Anna Korent, raziskovalka strojnega učenja. V zadnjem času sem se poglobljeno ukvarjala z vašo metodo bipropagacije in vam najprej iskreno čestitam za idejo — ravno zato, ker se mi je zdela vredna resne obravnave, sem jo neodvisno reproducirala in nad njo izvedla niz kontroliranih eksperimentov na MNIST in CIFAR-10. Pišem vam kolegialno, v duhu, da bi na vašem delu gradila in mu dala vidnost, ne pa da bi ga rušila.

Najprej dobra novica, ki je zame osrednja: **vaša temeljna intuicija — da nadzor na ravni posamezne plasti pomaga pri učenju globokih mrež — se v mojih eksperimentih potrjuje.** Plastno/lokalno učenje je konkurenčno in robustno na globino; pri globini 16 ostane stabilno (lokalna izguba 0.9685), medtem ko se navadni globoki backprop sesuje. Vaš požrešni "scaffold" (plast za plastjo) se je izkazal za trden temelj — to je pozitiven, publikabilen rezultat.

Hkrati naj z vami iskreno delim nekaj opažanj, ki bi jih rada razumela skupaj z vami, in me res zanima vaš pogled:

- Specifičnih številk (npr. "25× hitreje", "~100 % zanesljivo") v moji postavitvi nisem uspela reproducirati proti dobro nastavljenemu sodobnemu backpropu (Adam + He + BatchNorm), kjer je imel ta najnižjo varianco med semeni.
- Pri repozitoriju z deterministično inicializacijo v svoji reimplementaciji nisem uspela doseči poročanih 100 % na MNIST; dobila sem okrog 88 % (pri m≥16 nevronih/razred). Iskreno bi me zanimalo, kje se najini postavitvi razhajata — z veseljem delim svojo kodo, da primerjava postane lažja.
- Moja kontrolirana dekompozicija nakazuje, da je delujoči mehanizem predvsem nadzor na ravni plasti, in ne odsotnost backpropa kot taka: deeply-supervised kontrola (pomožne glave + globalni gradient) se ujema z lokalno izgubo na vsaki globini (globina 16: 0.9684 proti 0.9685). To se mi zdi zanimiva ugotovitev, ob kateri bi zelo cenila vaše mnenje.

Z veseljem vam dam na voljo celoten repozitorij in osnutek članka [povezava do repozitorija — bom dodala ob objavi]. Predvsem pa me zanima vaše mnenje, morebitni popravki in — če bi vas to veselilo — tudi sodelovanje. Povsem mogoče je, da kak del vaše metode v svoji rekonstrukciji nisem zajela natančno (npr. vaše pravilo za vmesne cilje pri večrazrednih problemih, ki v javni kodi ni dosegljivo), in bi to z veseljem popravila.

Najlepša hvala za vaš čas in za navdihujočo idejo.

S spoštovanjem,
Anna Korent
