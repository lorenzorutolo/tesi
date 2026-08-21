# -*- coding: utf-8 -*-
"""Pipeline di analisi delle campagne di benchmark (prototipo locale).

Nasce come esportazione del notebook Colab condiviso col relatore
    https://colab.research.google.com/drive/1S9zjz6AGpKScwIXG_ngGuoyQz1NVYbNO
e ne riprende il nucleo numerico (entropia, KL simmetrizzata, descrittori
bayesiani e credali, curve accuracy-rejection, confronto fra modelli).

Differenze volute rispetto al Colab, tutte annotate nel punto in cui compaiono:
  - le distribuzioni vengono rinormalizzate a somma 1 prima di ogni calcolo;
  - le KL su tutte le coppie sono vettorizzate, altrimenti le 120 permutazioni
    di CommonsenseQA costerebbero minuti per campagna;
  - lo script gira su piu' CSV in una volta sola e produce anche le tabelle
    riassuntive e il confronto fra modelli, invece di una figura per file.

Regime di normalizzazione: si lavora sempre "con altro", cioe' sulle K classi
canoniche piu' lo stato residuo, con il logaritmo in base K+1 (3 classi per
BoolQ, 6 per CommonsenseQA). E' la convenzione del paper del relatore e quella
adottata nel Capitolo 6 della tesi.

Uso tipico:
    python backend/generatore_grafici.py                  # le otto campagne
    python backend/generatore_grafici.py CSV_1 CSV_2 ...  # campagne scelte
    python backend/generatore_grafici.py --x=10           # solo 10 varianti

[1] Import librerie
"""

import csv
import glob
import json
import math
import os
import re
import sys
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except ImportError:
    files = None
    IN_COLAB = False

# In locale: ogni plt.show() salva un PNG; a fine script genero una galleria
# HTML con sidebar (click / frecce su-giu') e la apro nel browser.
if not IN_COLAB:
    import atexit
    import webbrowser
    import matplotlib
    matplotlib.use("Agg")

    _OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "grafici_output")
    os.makedirs(_OUT_DIR, exist_ok=True)
    for _f in os.listdir(_OUT_DIR):
        if _f.endswith((".png", ".html")):
            try:
                os.remove(os.path.join(_OUT_DIR, _f))
            except OSError:
                pass

    _figures_meta = []  # [(filename, titolo, gruppo)]

    def _estrai_titolo(fig):
        for ax in fig.get_axes():
            t = ax.get_title()
            if t:
                return t.replace("\n", " ")
        return ""

    def _show_and_save(*args, **kwargs):
        n = len(_figures_meta) + 1
        filename = f"figura_{n:02d}.png"
        path = os.path.join(_OUT_DIR, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        titolo = _estrai_titolo(plt.gcf()) or f"Figura {n}"
        _figures_meta.append((filename, titolo, globals().get("GRUPPO", "Grafici")))
        print(f"  [{n:02d}] {titolo}")
        plt.close("all")

    plt.show = _show_and_save

    # Galleria a confronto: 1/2/4 pannelli affiancati, sidebar con le figure
    # raggruppate per tema, badge colorati = in quale pannello sta ogni figura.
    _TEMPLATE_GALLERIA = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<title>Grafici Benchmark</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,sans-serif;display:flex;height:100vh;background:#eceff1}
  aside{width:330px;border-right:1px solid #ddd;overflow-y:auto;background:#fafafa;flex-shrink:0}
  aside h1{font-size:13px;margin:0;padding:12px 14px;border-bottom:1px solid #ddd;background:#fff}
  .layouts{display:flex;gap:6px;padding:10px 14px;border-bottom:1px solid #ddd;background:#fff;align-items:center}
  .layouts span{font-size:11px;color:#888;margin-right:4px}
  .layouts button{width:34px;height:26px;border:1px solid #bbb;background:#fff;border-radius:4px;cursor:pointer;font-weight:bold}
  .layouts button.sel{background:#2196F3;border-color:#2196F3;color:#fff}
  details{border-bottom:1px solid #eee}
  summary{padding:8px 14px;font-size:12px;font-weight:bold;cursor:pointer;background:#f0f0f0;user-select:none}
  ul{list-style:none;margin:0;padding:0}
  .item{display:flex;align-items:flex-start;gap:6px;padding:7px 10px 7px 14px;cursor:pointer;border-bottom:1px solid #f2f2f2;font-size:12px;line-height:1.3}
  .item:hover{background:#e3f2fd}
  .item .num{color:#999;font-weight:bold}
  .item .tit{flex:1}
  .item .badges{display:flex;gap:3px}
  .badge{width:16px;height:16px;border-radius:50%;color:#fff;font-size:10px;font-weight:bold;display:flex;align-items:center;justify-content:center}
  .b0{background:#2196F3}.b1{background:#FF9800}.b2{background:#4CAF50}.b3{background:#9C27B0}
  main{flex:1;display:grid;gap:8px;padding:8px;min-width:0}
  main[data-layout="1"]{grid-template-columns:1fr;grid-template-rows:1fr}
  main[data-layout="2"]{grid-template-columns:1fr 1fr;grid-template-rows:1fr}
  main[data-layout="4"]{grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr}
  .pannello{border:2px solid #cfd8dc;border-radius:6px;background:#fff;display:flex;flex-direction:column;overflow:hidden;cursor:pointer;min-height:0;min-width:0}
  .pannello.attivo{box-shadow:0 0 0 2px currentColor}
  .pannello .barra{display:flex;align-items:center;gap:8px;padding:6px 10px;border-bottom:1px solid #eee}
  .pannello .tag{font-size:10px;font-weight:bold;color:#fff;padding:2px 7px;border-radius:10px}
  .pannello h2{font-size:12px;margin:0;flex:1;font-weight:600;color:#37474F}
  .pannello img{flex:1;min-height:0;width:100%;object-fit:contain;background:#fff}
  .hint{position:fixed;bottom:6px;right:12px;font-size:11px;color:#78909c;background:#fff;padding:3px 8px;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.15)}
</style></head><body>
<aside>
  <h1>Grafici (__N__) &mdash; __CSV__</h1>
  <div class="layouts"><span>Pannelli:</span>
    <button data-n="1">1</button><button data-n="2" class="sel">2</button><button data-n="4">4</button>
  </div>
  <div id="gruppi">__SEZIONI__</div>
</aside>
<main id="pannelli" data-layout="2"></main>
<div class="hint">clic pannello = attiva &middot; clic figura = assegna &middot; &uarr;&darr; figura &middot; &larr;&rarr; pannello &middot; 1/2/4 layout &middot; doppio clic = apri PNG</div>
<script>
const FIGURE = __DATA__;
const COLORI = ['#2196F3','#FF9800','#4CAF50','#9C27B0'];
let layout = 2, attivo = 0;
const assegnate = [0,1,2,3].map(i => Math.min(i, FIGURE.length-1));
const main = document.getElementById('pannelli');
const pannelli = [];
for (let i=0;i<4;i++){
  const p = document.createElement('section');
  p.className = 'pannello';
  p.style.color = COLORI[i];
  p.innerHTML = '<div class="barra"><span class="tag" style="background:'+COLORI[i]+'">'+(i+1)+
                '</span><h2></h2></div><img alt="">';
  p.addEventListener('click', ()=>{ attivo = i; render(); });
  p.addEventListener('dblclick', ()=>window.open(FIGURE[assegnate[i]].fn));
  main.appendChild(p); pannelli.push(p);
}
const items = [...document.querySelectorAll('.item')];
items.forEach(it => it.addEventListener('click', ()=>{ assegnate[attivo] = +it.dataset.i; render(); }));
function render(){
  main.dataset.layout = layout;
  if (attivo >= layout) attivo = 0;
  pannelli.forEach((p,i)=>{
    p.style.display = i < layout ? '' : 'none';
    p.classList.toggle('attivo', i === attivo);
    const f = FIGURE[assegnate[i]];
    p.querySelector('h2').textContent = f.titolo;
    p.querySelector('img').src = f.fn;
  });
  items.forEach(it=>{ it.querySelector('.badges').innerHTML = ''; });
  for (let i=0;i<layout;i++){
    const it = items[assegnate[i]];
    if (!it) continue;
    const b = document.createElement('span');
    b.className = 'badge b'+i; b.textContent = i+1;
    it.querySelector('.badges').appendChild(b);
  }
}
document.querySelectorAll('.layouts button').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    layout = +btn.dataset.n;
    document.querySelectorAll('.layouts button').forEach(b=>b.classList.toggle('sel', b===btn));
    render();
  });
});
document.addEventListener('keydown', e=>{
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp'){
    e.preventDefault();
    const d = e.key === 'ArrowDown' ? 1 : -1;
    assegnate[attivo] = (assegnate[attivo] + d + FIGURE.length) % FIGURE.length;
    render();
    const it = items[assegnate[attivo]];
    if (it) it.scrollIntoView({block:'nearest'});
  } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft'){
    const d = e.key === 'ArrowRight' ? 1 : -1;
    attivo = (attivo + d + layout) % layout;
    render();
  } else if (['1','2','4'].includes(e.key)){
    layout = +e.key;
    document.querySelectorAll('.layouts button').forEach(b=>b.classList.toggle('sel', b.dataset.n===e.key));
    render();
  }
});
render();
</script></body></html>"""

    def _genera_galleria():
        if not _figures_meta:
            return
        dati = json.dumps([{"fn": fn, "titolo": t} for fn, t, _g in _figures_meta],
                          ensure_ascii=False)
        gruppi = {}
        for i, (fn, t, g) in enumerate(_figures_meta):
            gruppi.setdefault(g, []).append((i, t))
        sezioni = []
        for g, voci in gruppi.items():
            righe = "\n".join(
                f'<li class="item" data-i="{i}"><span class="num">{i + 1:02d}</span>'
                f'<span class="tit">{t}</span><span class="badges"></span></li>'
                for i, t in voci)
            sezioni.append(f'<details open><summary>{g} ({len(voci)})</summary>'
                           f'<ul>{righe}</ul></details>')
        percorsi = globals().get("PERCORSI", [])
        csv_nome = (f"{len(percorsi)} campagne" if len(percorsi) != 1
                    else os.path.basename(str(percorsi[0])))
        html = (_TEMPLATE_GALLERIA
                .replace("__CSV__", csv_nome)
                .replace("__N__", str(len(_figures_meta)))
                .replace("__SEZIONI__", "\n".join(sezioni))
                .replace("__DATA__", dati))
        index_path = os.path.join(_OUT_DIR, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)
        url = "file:///" + os.path.abspath(index_path).replace("\\", "/")
        print(f"\nGalleria pronta: {url}")
        # La galleria e' sempre lo stesso file: index.html viene riscritto a
        # ogni esecuzione, quindi basta ricaricare la scheda gia' aperta. Con
        # --no-apri il browser non viene richiamato, cosi' piu' esecuzioni di
        # fila non riempiono lo schermo di schede.
        if "--no-apri" not in sys.argv:
            webbrowser.open(url)

    atexit.register(_genera_galleria)

"""[2] Campagne da analizzare

Senza argomenti lo script prende tutti i ``test/<modello>/risultati_*.csv``
della radice del progetto, cioe' le otto campagne (4 modelli x 2 dataset).
Con argomenti prende i path passati a mano.

Opzioni:
    --x=N               tiene solo l'originale + le prime N varianti di ogni
                        domanda (le varianti sono in ordine seedato, quindi
                        "le prime N" e' riproducibile). Serve al confronto
                        10 vs 30 parafrasi.
    --solo-filtrato     esegue solo il regime con il filtro di convergenza
    --solo-non-filtrato esegue solo il regime senza filtro
    --no-apri           scrive la galleria senza aprirla nel browser
    --tesi              scrive anche le tre figure del Capitolo 6 in
                        versione stampa, in Documentazione/latex/immagini/
    --tesi-extra        con --tesi, aggiunge le cinque figure che nel capitolo
                        sono rese come tabelle o prosa (accuratezza a barre,
                        heatmap di Pearson, aree, coppia a KL massima, effetto
                        del filtro)

La galleria e' sempre lo stesso file, grafici_output/index.html: ogni
esecuzione lo riscrive da capo insieme ai PNG, quindi la pagina e' una sola e
si aggiorna ricaricandola.
"""

RADICE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PREFISSI_DATASET = [
    ("boolq_par_", "BoolQ"),
    ("commonsenseqa_perm_", "CommonsenseQA"),
]


def etichetta_modello(path):
    """Nome del modello ricavato dal nome del file dei risultati."""
    base = re.sub(r"\.csv$", "", os.path.basename(path))
    base = re.sub(r"^risultati_", "", base)
    for prefisso, _ in PREFISSI_DATASET:
        if base.startswith(prefisso):
            return base[len(prefisso):]
    return base


def nome_dataset(path):
    base = os.path.basename(path)
    for prefisso, nome in PREFISSI_DATASET:
        if prefisso in base:
            return nome
    return "sconosciuto"


_argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
_opzioni = [a for a in sys.argv[1:] if a.startswith("--")]

X_VARIANTI = None
for _o in _opzioni:
    if _o.startswith("--x="):
        X_VARIANTI = int(_o.split("=", 1)[1])

# Filtro di convergenza: tiene solo le ripetizioni che concordano con l'argmax
# della domanda originale. E' il flag FILTERED del Colab; qui si generano per
# default entrambe le versioni, cosi' si vede l'effetto del filtro.
FILTRI = (False, True)
if "--solo-filtrato" in _opzioni:
    FILTRI = (True,)
elif "--solo-non-filtrato" in _opzioni:
    FILTRI = (False,)

if IN_COLAB:
    print("Seleziona i CSV con i risultati del benchmark:")
    uploaded = files.upload()
    PERCORSI = sorted(uploaded.keys())
elif _argomenti:
    PERCORSI = _argomenti
else:
    PERCORSI = sorted(glob.glob(os.path.join(RADICE, "test", "*", "risultati_*.csv")))

if not PERCORSI:
    print("ATTENZIONE: Nessun CSV trovato. Passa i path come argomenti oppure metti le "
          "campagne in test/<modello>/risultati_*.csv")
    sys.exit(1)

print(f"Campagne da analizzare ({len(PERCORSI)}):")
for _p in PERCORSI:
    print(f"  - {nome_dataset(_p):14s} {etichetta_modello(_p):22s} {_p}")
print(f"Filtro di convergenza: {', '.join('sì' if f else 'no' for f in FILTRI)}"
      f" | varianti tenute: {'tutte' if X_VARIANTI is None else X_VARIANTI}")

"""[3] Nucleo numerico (allineato al Colab condiviso col relatore)

Ogni domanda e' una matrice ``n x k``: una riga per ripetizione (la riga 0 e'
la domanda originale), una colonna per classe. Le colonne comprendono lo stato
residuo ``altro``, quindi k = K+1: e' il regime scelto per la tesi, e il
logaritmo in base k tiene entropie e divergenze in [0, 1].

Unica differenza voluta rispetto al Colab: le righe vengono rinormalizzate a
somma 1 (nei CSV la somma sta fra 0,997 e 1,000 per via della massa non
catturata dai top_logprobs). L'effetto sui valori e' nell'ordine di 1e-4 e non
cambia nessun ordinamento, ma rende l'entropia una vera entropia.

Il calcolo delle KL su tutte le coppie e' vettorizzato: con le 120
permutazioni di CommonsenseQA sono 7260 coppie per domanda, che in Python puro
costerebbero minuti per campagna.
"""

TOL = 1e-12


def entropie(P):
    """Entropia normalizzata di ogni riga di ``P`` (n x k), in [0, 1]."""
    k = P.shape[1]
    logP = np.where(P > TOL, np.log(np.where(P > TOL, P, 1.0)), 0.0)
    return -(P * logP).sum(axis=1) / math.log(k)


def entropia_vettore(p):
    """Entropia normalizzata di una singola distribuzione."""
    return float(entropie(np.asarray(p, dtype=float)[None, :])[0])


def kl2_coppie(P):
    """Matrice n x n delle KL simmetrizzate fra le righe di ``P``.

    D[i, j] = sum_c m_i m_j p_i(c) log(p_i(c) / p_j(c)) / log(k), dove m e' la
    maschera dei termini non nulli: i termini con p_i(c) = 0 o p_j(c) = 0
    contano zero, esattamente come nella KL scalare del Colab. La versione
    simmetrizzata e' (D + D^T) / 2.
    """
    k = P.shape[1]
    maschera = P > TOL
    Pm = np.where(maschera, P, 0.0)
    L = np.where(maschera, np.log(np.where(maschera, P, 1.0)), 0.0)
    A = Pm * L                       # p log p, gia' azzerato fuori maschera
    Mf = maschera.astype(float)
    D = (A @ Mf.T - Pm @ L.T) / math.log(k)
    return (D + D.T) / 2.0


# I nove descrittori richiesti dal relatore, piu' H_orig come baseline. Le
# chiavi AU_B / EU_B / AU_C / EU_C sono i quattro del paper e coincidono con
# quattro dei nove: la mappa e' nel commento a fianco.
NOMI_DESCRITTORI = [
    ("Hpmean", "[1] Hpmean (entropia della media)"),
    ("AU_B",   "[2] Hmean (media delle entropie) = $AU_B$"),
    ("AU_C",   "[3] Hmax (massima entropia) = $AU_C$"),
    ("Hmin",   "[4] Hmin (minima entropia)"),
    ("EU_C",   "[5] KLmax (max KL simmetrica) = $EU_C$"),
    ("EU_B",   "[6] Jensen (entropia della media - media delle entropie) = $EU_B$"),
    ("dH",     "[7] Hmax - Hmin (Delta Entropia)"),
    ("Hmax_KL", "[8] Hmax - KLmax"),
    ("Hmin_KL", "[9] Hmin - KLmax"),
    ("H_orig", "[10] H_orig (entropia della sola domanda originale)"),
]


def descrittori_da_matrice(P, idx_corretta):
    """Descrittori bayesiani e credali di UN insieme di distribuzioni.

    Vale con qualunque n >= 2: le righe sono le ripetizioni di un modello
    oppure, nel confronto fra modelli, le risposte originali dei vari modelli
    alla stessa domanda.
    """
    ents = entropie(P)
    p_media = P.mean(axis=0)
    h_media = entropia_vettore(p_media)
    au_b = float(ents.mean())
    au_c = float(ents.max())
    h_min = float(ents.min())
    kl = kl2_coppie(P)
    eu_c = float(kl[np.triu_indices(len(P), k=1)].max())
    return {
        "Hpmean": h_media,
        "AU_B": au_b,
        "AU_C": au_c,
        "Hmin": h_min,
        "EU_C": eu_c,
        "EU_B": h_media - au_b,
        "dH": au_c - h_min,
        "Hmax_KL": au_c - eu_c,
        "Hmin_KL": h_min - eu_c,
        # Entropia della sola risposta originale: il segnale che si avrebbe
        # senza ripetizioni. Baseline per capire se le varianti aggiungono
        # qualcosa rispetto alla domanda secca.
        "H_orig": float(ents[0]),
        "Pred": int(idx_corretta == int(np.argmax(p_media))),
    }


def taglia_matrice(P, x):
    """Originale (riga 0) + le prime ``x`` varianti; ``x=None`` = tutte."""
    return P if x is None else P[:x + 1]


def filtra_matrice(P):
    """Filtro di convergenza: tiene le ripetizioni che concordano con l'argmax
    della domanda originale (riga 0). La riga 0 sopravvive sempre."""
    riferimento = int(np.argmax(P[0]))
    return P[np.argmax(P, axis=1) == riferimento]


"""[4] Caricamento di una campagna

Una campagna sta in memoria una alla volta: i CSV di BoolQ sono da ~27 MB e
tenerne otto aperti insieme non serve. Del passaggio si conservano solo i dati
leggeri che servono al confronto fra modelli della sezione [7], cioe' la
distribuzione della risposta originale e la media sulle varianti, per id.
"""


class Campagna:
    def __init__(self, path):
        self.path = path
        self.modello = etichetta_modello(path)
        self.dataset = nome_dataset(path)
        self.classi = []          # classi valide + "altro" in coda
        self.classi_valide = []
        self.righe = []           # id, reale, idx_reale, matrice, testi

    @property
    def etichetta(self):
        return f"{self.modello} — {self.dataset}"


def carica_campagna(path):
    camp = Campagna(path)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            alternative = json.loads(row["alternative_json"])
            if not alternative:
                continue
            if not camp.classi:
                chiavi = list(alternative[0].get("probabilita", {}).keys())
                camp.classi_valide = [c for c in chiavi if c != "altro"]
                camp.classi = camp.classi_valide + ["altro"]
            reale = row["reale"].strip()
            if reale not in camp.classi:
                continue
            M = np.array([[a.get("probabilita", {}).get(c, 0.0) for c in camp.classi]
                          for a in alternative], dtype=float)
            somme = M.sum(axis=1, keepdims=True)
            M = np.divide(M, somme, out=np.zeros_like(M), where=somme > TOL)
            camp.righe.append({
                "id": row["id"],
                "domanda": row["domanda"],
                "reale": reale,
                "idx_reale": camp.classi.index(reale),
                "matrice": M,
                "testi": [a.get("domanda_alt", "") for a in alternative],
            })
    return camp


def conta_varianti(path):
    """Numero di varianti per domanda, letto dalla prima riga utile del CSV."""
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            alternative = json.loads(row["alternative_json"])
            if alternative:
                return len(alternative)
    return 0


def prepara(riga, filtrato):
    """Matrice di una domanda dopo taglio e filtro; None se resta troppo poco."""
    M = taglia_matrice(riga["matrice"], X_VARIANTI)
    if filtrato:
        M = filtra_matrice(M)
    return M if len(M) >= 2 else None


"""[5] Curve accuracy-rejection

Protocollo del relatore: si ordinano le domande per descrittore decrescente,
si rifiuta una frazione crescente delle piu' incerte e si misura l'accuratezza
sulle domande tenute. La banda grigia e' la baseline a rifiuto casuale.
"""

FRAZIONI_RIFIUTATE = np.linspace(0.0, 0.95, 20)
N_ORDINAMENTI_CASUALI = 200

DESCRITTORI_PAPER = [
    ("$AU_B$", "AU_B", "#1f77b4", "solid"),
    ("$EU_B$", "EU_B", "#1f77b4", "dotted"),
    ("$AU_C$", "AU_C", "#ff7f0e", "solid"),
    ("$EU_C$", "EU_C", "#ff7f0e", "dotted"),
    ("$H$ (orig.)", "H_orig", "#2ca02c", "solid"),
]


def accuratezza_su_tenute(esiti_ordinati):
    cumulate = np.cumsum(esiti_ordinati)
    return np.asarray([cumulate[max(1, int(round((1 - f) * len(esiti_ordinati)))) - 1]
                       / max(1, int(round((1 - f) * len(esiti_ordinati))))
                       for f in FRAZIONI_RIFIUTATE])


def grafico_accuracy_rejection(descr, titolo, descrittori=DESCRITTORI_PAPER,
                               figsize=(8, 6)):
    """Curva accuracy-rejection di una lista di dict di descrittori."""
    esiti = np.asarray([d["Pred"] for d in descr], dtype=float)
    if len(esiti) < 2:
        print(f"ATTENZIONE: Dati insufficienti per: {titolo}")
        return None
    print(f"  {titolo}: n={len(esiti)} domande, "
          f"accuratezza globale {esiti.mean():.4f}")

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    rng = np.random.default_rng(42)
    curve = np.asarray([accuratezza_su_tenute(rng.permutation(esiti))
                        for _ in range(N_ORDINAMENTI_CASUALI)])
    ax.fill_between(FRAZIONI_RIFIUTATE,
                    curve.mean(axis=0) - curve.std(axis=0),
                    curve.mean(axis=0) + curve.std(axis=0),
                    color="black", alpha=0.25, linewidth=0)
    ax.plot(FRAZIONI_RIFIUTATE, curve.mean(axis=0), color="black",
            linewidth=1.6, label="casuale")

    # Area sotto la curva, come sintesi numerica di "quanto in alto sta".
    # Serve a ordinare i descrittori senza doverli confrontare a occhio: e' la
    # media dell'accuratezza sulle frazioni rifiutate, quindi si legge nella
    # stessa unita' dell'asse y. Il guadagno e' rispetto alla baseline casuale,
    # che per costruzione resta piatta sull'accuratezza globale.
    area_casuale = float(curve.mean(axis=0).mean())
    aree = []
    curve_per_chiave = {}
    aree_per_chiave = {}
    for etichetta, chiave, colore, stile in descrittori:
        valori = np.asarray([d[chiave] for d in descr])
        ordine = np.argsort(valori, kind="stable")
        curva = accuratezza_su_tenute(esiti[ordine])
        aree.append((float(curva.mean()), etichetta))
        curve_per_chiave[chiave] = curva
        aree_per_chiave[chiave] = float(curva.mean())
        ax.plot(FRAZIONI_RIFIUTATE, curva,
                color=colore, linestyle=stile, linewidth=1.8, label=etichetta)

    print(f"    area sotto la curva (casuale {area_casuale:.4f}): "
          + "  ".join(f"{e.replace('$', '')} {a:.4f} ({a - area_casuale:+.4f})"
                      for a, e in sorted(aree, reverse=True)))

    ax.set_title(titolo, pad=15, fontsize=11)
    ax.set_xlabel("Frazione di domande rifiutate (dal descrittore più alto)")
    ax.set_ylabel("Accuratezza sulle domande tenute (maggioranza)")
    ax.set_xlim(0, 0.95)
    ax.grid(linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

    # Gli stessi numeri servono alle figure per la tesi della sezione [10]:
    # curve, banda casuale e aree si restituiscono invece di ricalcolarli.
    return {
        "curve": curve_per_chiave,
        "aree": aree_per_chiave,
        "area_casuale": area_casuale,
        "casuale": curve.mean(axis=0),
        "casuale_min": curve.mean(axis=0) - curve.std(axis=0),
        "casuale_max": curve.mean(axis=0) + curve.std(axis=0),
        "accuratezza": float(esiti.mean()),
        "domande": int(len(esiti)),
    }


"""[6] Passaggio principale: una campagna alla volta

Per ogni campagna e per ogni regime di filtro si calcolano i descrittori di
tutte le domande, da cui discendono la curva accuracy-rejection, la colonna
della tabella di Pearson e i conteggi di accuratezza. Le tabelle riassuntive
si disegnano dopo, quando tutte le campagne sono state viste.
"""

# Accumulatori leggeri, sopravvivono alla singola campagna
accuratezze = {}        # (path, filtrato) -> dict con i tre livelli
confusioni = {}         # (path, filtrato) -> dict TP/TN/FP/FN (solo binari)
pearson = {}            # (path, filtrato) -> {chiave_descrittore: r}
originali_per_modello = {}   # dataset -> {modello: {id: vettore riga 0}}
medie_per_modello = {}       # dataset -> {modello: {id: vettore medio}}
classi_per_dataset = {}      # dataset -> lista delle classi (ordine canonico)
coppie_max_kl = {}      # (path, filtrato) -> (kl, riga, i, j)
dati_curve = {}         # (path, filtrato) | ("modelli", dataset) -> curve e aree
au_eu_modelli = {}      # dataset -> (au_orig, eu_orig, au_medie, eu_medie, ...)

# Le varianti minime comuni a un dataset servono per la media sulle parafrasi
# del confronto fra modelli: mediare su 30 parafrasi per un modello e su 10 per
# un altro non e' un confronto alla pari.
varianti_per_campagna = {p: conta_varianti(p) for p in PERCORSI}
minimo_varianti = {}
_varianti_viste = {}
for _p in PERCORSI:
    d = nome_dataset(_p)
    minimo_varianti[d] = min(minimo_varianti.get(d, 10 ** 9),
                             varianti_per_campagna[_p])
    _varianti_viste.setdefault(d, set()).add(varianti_per_campagna[_p])

for d, viste in _varianti_viste.items():
    if len(viste) > 1:
        print(f"\nATTENZIONE:  Su {d} le campagne NON hanno lo stesso numero di varianti "
              f"({sorted(viste)}).")
        print(f"   La media sulle varianti del confronto fra modelli viene "
              f"troncata a {minimo_varianti[d]}, cosi' il confronto resta alla "
              f"pari; le curve accuracy-rejection per campagna usano invece "
              f"tutte le varianti disponibili di quella campagna.")
        for _p in PERCORSI:
            if nome_dataset(_p) == d:
                print(f"     {etichetta_modello(_p):22s} "
                      f"{varianti_per_campagna[_p]:4d} varianti")

for percorso in PERCORSI:
    camp = carica_campagna(percorso)
    print(f"\n{'=' * 78}\n{camp.etichetta}  ({len(camp.righe)} domande, "
          f"classi {camp.classi}, {varianti_per_campagna[percorso]} varianti)")

    dataset = camp.dataset
    if dataset in classi_per_dataset and classi_per_dataset[dataset] != camp.classi:
        print(f"ATTENZIONE: Classi diverse dentro lo stesso dataset: "
              f"{classi_per_dataset[dataset]} vs {camp.classi}")
    classi_per_dataset.setdefault(dataset, camp.classi)

    # Dati leggeri per la sezione [7], calcolati una volta sola: non dipendono
    # dal filtro, perche' la riga 0 sopravvive sempre al filtro. La media si fa
    # sul numero di varianti minimo comune al dataset (e sul taglio --x, se
    # c'e'): mediare su 30 parafrasi per un modello e su 10 per un altro non
    # sarebbe un confronto alla pari.
    n_comune = minimo_varianti[dataset]
    if X_VARIANTI is not None:
        n_comune = min(n_comune, X_VARIANTI + 1)
    originali_per_modello.setdefault(dataset, {})[camp.modello] = {
        r["id"]: (r["matrice"][0], r["reale"]) for r in camp.righe}
    medie_per_modello.setdefault(dataset, {})[camp.modello] = {
        r["id"]: (r["matrice"][:n_comune].mean(axis=0), r["reale"])
        for r in camp.righe}

    for filtrato in FILTRI:
        tag = "filtrato" if filtrato else "non filtrato"
        # Raggruppate per regime e non per campagna: nella sidebar della
        # galleria le otto curve dello stesso regime stanno insieme, che e' il
        # confronto che interessa. Il modello e il dataset sono nel titolo.
        GRUPPO = f"Accuracy-rejection ({tag})"

        descr = []
        n_orig = n_orig_ok = n_var = n_var_ok = 0
        tp = tn = fp = fn = 0
        migliore = None
        scartate = 0
        for riga in camp.righe:
            M = prepara(riga, filtrato)
            if M is None:
                scartate += 1
                continue
            idx = riga["idx_reale"]
            d = descrittori_da_matrice(M, idx)
            descr.append(d)

            # Accuratezza ai tre livelli, tutta derivata dalle distribuzioni:
            # la risposta e' l'argmax, "altro" compreso (se vince "altro" la
            # risposta e' sbagliata, che e' il comportamento voluto).
            n_orig += 1
            n_orig_ok += int(np.argmax(M[0]) == idx)
            predette = np.argmax(M[1:], axis=1)
            n_var += len(predette)
            n_var_ok += int((predette == idx).sum())

            if len(camp.classi_valide) == 2:
                vero = riga["reale"] == camp.classi_valide[0]
                predetto = int(np.argmax(M.mean(axis=0))) == camp.classi.index(
                    camp.classi_valide[0])
                if predetto and vero:
                    tp += 1
                elif not predetto and not vero:
                    tn += 1
                elif predetto and not vero:
                    fp += 1
                else:
                    fn += 1

            # Coppia con la KL massima: serve alla tabella qualitativa
            kl = kl2_coppie(M)
            i, j = np.unravel_index(np.argmax(np.triu(kl, k=1)), kl.shape)
            if migliore is None or kl[i, j] > migliore[0]:
                migliore = (float(kl[i, j]), riga, int(i), int(j), M)

        if not descr:
            print(f"ATTENZIONE: Nessuna domanda utilizzabile ({tag}).")
            continue

        n_magg = len(descr)
        n_magg_ok = sum(d["Pred"] for d in descr)
        accuratezze[(percorso, filtrato)] = {
            "originale": 100.0 * n_orig_ok / n_orig if n_orig else 0.0,
            "varianti": 100.0 * n_var_ok / n_var if n_var else 0.0,
            "maggioranza": 100.0 * n_magg_ok / n_magg,
            "domande": n_magg,
            "scartate": scartate,
        }
        print(f"  [{tag}] accuratezza  originale {100.0 * n_orig_ok / max(n_orig, 1):.2f}%"
              f"  varianti {100.0 * n_var_ok / max(n_var, 1):.2f}%"
              f"  maggioranza {100.0 * n_magg_ok / n_magg:.2f}%"
              + (f"  (domande scartate: {scartate})" if scartate else ""))

        if len(camp.classi_valide) == 2:
            confusioni[(percorso, filtrato)] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            print(f"  [{tag}] confusione (maggioranza, positiva = "
                  f"'{camp.classi_valide[0]}')  TP={tp} TN={tn} FP={fp} FN={fn}"
                  f"  precision={prec:.4f} recall={rec:.4f}")

        # Pearson di ogni descrittore contro la scorrettezza (0 = corretta)
        errata = [1 - d["Pred"] for d in descr]
        pearson[(percorso, filtrato)] = {}
        for chiave, _nome in NOMI_DESCRITTORI:
            valori = [d[chiave] for d in descr]
            if len(set(valori)) < 2:
                pearson[(percorso, filtrato)][chiave] = float("nan")
                continue
            pearson[(percorso, filtrato)][chiave] = float(
                stats.pearsonr(valori, errata).statistic)

        coppie_max_kl[(percorso, filtrato)] = migliore

        # Unica curva prevista: i quattro descrittori del paper piu' H_orig.
        # Gli altri cinque dei nove del relatore restano nella tabella di
        # Pearson, che li confronta senza costare una figura a testa.
        dati_curve[(percorso, filtrato)] = grafico_accuracy_rejection(
            descr, f"Accuracy-rejection — {camp.etichetta} ({tag})")

    del camp

"""[7] Confronto fra modelli

Due letture, entrambe con l'insieme credale formato dai MODELLI invece che
dalle ripetizioni: la curva accuracy-rejection sulle risposte originali e la
scomposizione AU/EU a barre. Le domande sono allineate per id
sull'intersezione dei modelli.
"""


def matrici_multimodello(per_modello, classi):
    """(descrittori, n_modelli) con una riga per modello sulla stessa domanda."""
    modelli = sorted(per_modello)
    if len(modelli) < 2:
        return [], modelli
    id_comuni = set.intersection(*[set(per_modello[m]) for m in modelli])
    righe = []
    for id_domanda in sorted(id_comuni):
        voci = [per_modello[m][id_domanda] for m in modelli]
        P = np.asarray([v[0] for v in voci], dtype=float)
        idx = classi.index(voci[0][1])
        righe.append(descrittori_da_matrice(P, idx))
    return righe, modelli


def barre_au_eu(descr):
    """(AU, EU) medi sulle domande. Le due componenti sono additive per
    costruzione: la loro somma e' l'entropia della distribuzione media."""
    au = float(np.mean([d["AU_B"] for d in descr]))
    eu = float(np.mean([d["EU_B"] for d in descr]))
    return au, eu


GRUPPO = "Confronto fra modelli"

for dataset, per_modello in sorted(originali_per_modello.items()):
    classi = classi_per_dataset[dataset]

    descr_orig, modelli = matrici_multimodello(per_modello, classi)
    if not descr_orig:
        print(f"ATTENZIONE: Servono almeno due modelli per il confronto su {dataset}.")
        continue
    descr_medie, _ = matrici_multimodello(medie_per_modello[dataset], classi)

    print(f"\n{'=' * 78}\nConfronto fra {len(modelli)} modelli su {dataset}: "
          f"{', '.join(modelli)}")
    print(f"  domande in comune: {len(descr_orig)} | media sulle parafrasi "
          f"calcolata su {minimo_varianti[dataset]} varianti (minimo comune)")

    # Niente curva H_orig: qui ogni riga e' gia' la risposta originale di un
    # modello, non esiste una singola "domanda secca".
    dati_curve[("modelli", dataset)] = grafico_accuracy_rejection(
        descr_orig,
        f"Accuracy-rejection fra modelli — {dataset}, risposte originali",
        descrittori=[d for d in DESCRITTORI_PAPER if d[1] != "H_orig"])

    # Scomposizione AU/EU: due barre nella stessa figura
    au_o, eu_o = barre_au_eu(descr_orig)
    au_m, eu_m = barre_au_eu(descr_medie)
    print(f"  AU/EU (PMF = domanda originale)   AU_B={au_o:.4f} EU_B={eu_o:.4f} "
          f"totale={au_o + eu_o:.4f}")
    print(f"  AU/EU (PMF = media delle varianti) AU_B={au_m:.4f} EU_B={eu_m:.4f} "
          f"totale={au_m + eu_m:.4f}")
    au_eu_modelli[dataset] = (au_o, eu_o, au_m, eu_m, len(descr_orig),
                              len(modelli), len(classi))

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
    etichette = ["Domanda\noriginale", "Media delle\nvarianti"]
    valori_au = [au_o, au_m]
    valori_eu = [eu_o, eu_m]
    ax.bar(etichette, valori_au, width=0.42, color="#00008B", label="AU")
    ax.bar(etichette, valori_eu, width=0.42, bottom=valori_au, color="#B00000",
           label="EU")
    for x, (a, e) in enumerate(zip(valori_au, valori_eu)):
        ax.text(x, a / 2, f"{a:.3f}", ha="center", va="center", fontsize=9,
                color="white")
        ax.text(x, a + e / 2, f"{e:.3f}", ha="center", va="center", fontsize=9,
                color="white")
        ax.text(x, a + e + 0.012, f"{a + e:.3f}", ha="center", va="bottom",
                fontsize=9, color="#B00000")
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, max(0.6, max(a + e for a, e in zip(valori_au, valori_eu)) * 1.3))
    ax.set_ylabel("Incertezza (entropia normalizzata)")
    ax.set_title(f"Scomposizione AU/EU al variare del modello — {dataset}",
                 fontsize=11, pad=12)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax.yaxis.grid(True, linestyle="dotted", color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    fig.text(0.5, 0.015,
             f"{len(descr_orig)} domande, insieme credale di {len(modelli)} "
             f"modelli ({len(classi)} classi)",
             ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()

"""[8] Tabelle riassuntive su tutte le campagne

Le tre tabelle del Capitolo 6, ciascuna con una riga per campagna invece di una
figura per campagna. Oltre alla versione in galleria vengono stampate le righe
LaTeX gia' pronte da incollare in tesi.tex.
"""


def _breve(modello):
    """Nome del modello accorciato per stare nell'intestazione di colonna."""
    return re.sub(r"-(instruct|it)$", "", modello)


def _tabella(ax, celle, colonne, larghezze, evidenzia=(), font_intestazione=9):
    ax.axis("off")
    tab = ax.table(cellText=celle, colLabels=colonne, colWidths=larghezze,
                   cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.5)
    for (r_i, c_i), cella in tab.get_celld().items():
        if r_i == 0:
            cella.set_text_props(fontweight="bold", color="white",
                                 fontsize=font_intestazione)
            cella.set_facecolor("#37474F")
        else:
            if r_i % 2 == 0:
                cella.set_facecolor("#ECEFF1")
            if c_i == 0:
                cella._loc = "left"
            if (r_i - 1, c_i) in evidenzia:
                cella.set_text_props(fontweight="bold")
    return tab


def _it(x, cifre=2):
    """Numero con la virgola decimale, come nel resto della tesi."""
    return f"{x:.{cifre}f}".replace(".", ",")


GRUPPO = "Tabelle riassuntive"

for filtrato in FILTRI:
    tag = "filtrato" if filtrato else "non filtrato"
    voci = [(p, accuratezze[(p, filtrato)]) for p in PERCORSI
            if (p, filtrato) in accuratezze]
    if not voci:
        continue

    celle = [[etichetta_modello(p), nome_dataset(p), _it(a["originale"]),
              _it(a["varianti"]), _it(a["maggioranza"])] for p, a in voci]
    fig, ax = plt.subplots(figsize=(10, 1 + 0.45 * len(celle)), dpi=100)
    _tabella(ax, celle, ["Modello", "Dataset", "Originale", "Varianti",
                         "Maggioranza"], [0.28, 0.20, 0.17, 0.17, 0.18])
    ax.set_title(f"Accuratezza ai tre livelli ({tag})", pad=14)
    plt.show()

    print(f"\n--- LaTeX: accuratezza ai tre livelli ({tag}) ---")
    for p, a in voci:
        print(f"        {etichetta_modello(p)} & {nome_dataset(p)} & "
              f"{_it(a['originale'])} & {_it(a['varianti'])} & "
              f"{_it(a['maggioranza'])} \\\\")

    # Confusione: solo le campagne binarie
    voci_bin = [(p, confusioni[(p, filtrato)]) for p in PERCORSI
                if (p, filtrato) in confusioni]
    if voci_bin:
        celle = []
        for p, c in voci_bin:
            prec = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0
            rec = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
            celle.append([etichetta_modello(p), str(c["TP"]), str(c["TN"]),
                          str(c["FP"]), str(c["FN"]), _it(prec, 4), _it(rec, 4)])
        fig, ax = plt.subplots(figsize=(10, 1 + 0.45 * len(celle)), dpi=100)
        _tabella(ax, celle, ["Modello", "TP", "TN", "FP", "FN", "Precision",
                             "Recall"], [0.28, 0.11, 0.11, 0.11, 0.11, 0.14, 0.14])
        ax.set_title(f"Matrice di confusione, risposta di maggioranza ({tag})",
                     pad=14)
        plt.show()

        print(f"\n--- LaTeX: confusione BoolQ ({tag}) ---")
        for riga in celle:
            print("        " + " & ".join(riga) + " \\\\")

    # Pearson: dieci descrittori x una colonna per campagna
    colonne = [p for p in PERCORSI if (p, filtrato) in pearson]
    celle = []
    evidenzia = set()
    massimi = {}
    for c_i, p in enumerate(colonne, start=1):
        valori = {k: pearson[(p, filtrato)][k] for k, _ in NOMI_DESCRITTORI}
        migliore = max(valori, key=lambda k: abs(valori[k]) if valori[k] == valori[k] else -1)
        massimi[c_i] = migliore
    for r_i, (chiave, nome) in enumerate(NOMI_DESCRITTORI):
        riga = [nome]
        for c_i, p in enumerate(colonne, start=1):
            r = pearson[(p, filtrato)][chiave]
            riga.append("n/d" if r != r else f"{r:+.4f}".replace(".", ","))
            if massimi[c_i] == chiave:
                evidenzia.add((r_i, c_i))
        celle.append(riga)

    larghezza = 0.34
    fig, ax = plt.subplots(figsize=(4.5 + 1.5 * len(colonne), 6), dpi=100)
    _tabella(ax, celle,
             ["Descrittore"] + [f"{_breve(etichetta_modello(p))}\n{nome_dataset(p)}"
                                for p in colonne],
             [larghezza] + [(1 - larghezza) / len(colonne)] * len(colonne),
             evidenzia=evidenzia, font_intestazione=8)
    ax.set_title(f"Pearson descrittore vs scorrettezza della maggioranza ({tag})",
                 pad=14)
    fig.text(0.5, 0.03, "esito: 0 = corretta, 1 = errata — in grassetto il |r| "
             "massimo di ogni colonna", ha="center", fontsize=9, style="italic",
             color="#546E7A")
    plt.show()

    print(f"\n--- LaTeX: Pearson ({tag}) ---")
    for riga in celle:
        print("        " + " & ".join(riga) + " \\\\")

"""[9] La coppia di ripetizioni con la KL massima

Tabella qualitativa: le due formulazioni della stessa domanda che spostano di
piu' la distribuzione di risposta. Una per campagna e per regime di filtro.
"""

for percorso in PERCORSI:
    for filtrato in FILTRI:
        chiave = (percorso, filtrato)
        if chiave not in coppie_max_kl or coppie_max_kl[chiave] is None:
            continue
        tag = "filtrato" if filtrato else "non filtrato"
        GRUPPO = f"Coppia max KL ({tag})"
        kl, riga, i, j, M = coppie_max_kl[chiave]
        classi = classi_per_dataset[nome_dataset(percorso)]

        celle = []
        for indice in (i, j):
            testo = textwrap.fill(str(riga["testi"][indice]), width=60)
            probs = [_it(M[indice][c], 4) for c in range(len(classi))]
            celle.append([f"#{indice}", testo] + probs
                         + [_it(float(entropie(M[indice:indice + 1])[0]), 4)])

        fig, ax = plt.subplots(figsize=(4 + 1.1 * len(classi), 4), dpi=100)
        _tabella(ax, celle, ["Rip.", "Formulazione"] + list(classi) + ["H"],
                 [0.07, 0.45] + [0.38 / (len(classi) + 1)] * (len(classi) + 1))
        ax.set_title(f"Coppia a KL massima — {etichetta_modello(percorso)}, "
                     f"{nome_dataset(percorso)} ({tag})\n"
                     f"KLsym = {kl:.4f} — risposta attesa: {riga['reale']}",
                     pad=14, fontsize=11)
        plt.show()

"""[10] Figure per la tesi (opzione --tesi)

Le figure del Capitolo 6 di Documentazione/latex/tesi.tex. La matematica non
viene rifatta: curve, aree e banda casuale arrivano da ``dati_curve``, riempito
nelle sezioni [6] e [7]. Cambia il disegno, pensato per la carta:

  - nessun titolo dentro l'immagine, perche' lo fornisce la caption LaTeX;
  - una sola legenda condivisa nelle griglie a pannelli;
  - font serif e corpo piu' grande, perche' in stampa l'immagine viene
    rimpicciolita a \\textwidth;
  - nomi di file parlanti invece di figura_NN.png.

Le figure finiscono in Documentazione/latex/immagini/, la cartella da caricare
su Overleaf accanto a tesi.tex.
"""

if "--tesi" in _opzioni and not IN_COLAB:
    DIR_TESI = os.path.join(RADICE, "Documentazione", "latex", "immagini")
    os.makedirs(DIR_TESI, exist_ok=True)
    print(f"\n{'=' * 78}\nFigure per la tesi -> {os.path.relpath(DIR_TESI, RADICE)}")

    STILE_TESI = {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": False,
        "savefig.dpi": 200,
    }

    COLORI_LIVELLI = [("originale", "Originale", "#3C6E9F"),
                      ("varianti", "Varianti", "#D08A45"),
                      ("maggioranza", "Maggioranza", "#5B8C5A")]

    ETICHETTE_HEATMAP = [
        ("Hpmean", "$H(\\bar{p})$  entropia della media"),
        ("AU_B", "$H_{mean} = AU_B$"),
        ("AU_C", "$H_{max} = AU_C$"),
        ("Hmin", "$H_{min}$"),
        ("EU_C", "$KL_{max} = EU_C$"),
        ("EU_B", "Jensen $= EU_B$"),
        ("dH", "$H_{max} - H_{min}$"),
        ("Hmax_KL", "$H_{max} - KL_{max}$"),
        ("Hmin_KL", "$H_{min} - KL_{max}$"),
        ("H_orig", "$H$ della domanda originale"),
    ]

    TESI_EXTRA = "--tesi-extra" in _opzioni

    def salva_tesi(fig, nome, extra=False):
        # Le figure marcate extra non entrano nel capitolo: nel testo quei
        # contenuti sono tabelle o prosa. Si disegnano solo su richiesta.
        if extra and not TESI_EXTRA:
            plt.close(fig)
            return
        percorso = os.path.join(DIR_TESI, nome)
        fig.savefig(percorso, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {nome}")

    def campagne_di(dataset):
        return sorted([p for p in PERCORSI if nome_dataset(p) == dataset],
                      key=etichetta_modello)

    DATASET_TESI = [d for d in ("BoolQ", "CommonsenseQA") if campagne_di(d)]

    def campagna_esempio(dataset, preferito="llama3.1"):
        """La campagna commentata per esteso nel testo, con ripiego se manca."""
        candidati = campagne_di(dataset)
        for p in candidati:
            if etichetta_modello(p).startswith(preferito):
                return p
        return candidati[0] if candidati else None

    with plt.rc_context(STILE_TESI):

        # --- 6.2 accuratezza ai tre livelli -------------------------------
        if accuratezze:
            fig, axes = plt.subplots(1, len(DATASET_TESI), figsize=(9.2, 3.9),
                                     sharey=True, squeeze=False)
            for ax, dataset in zip(axes[0], DATASET_TESI):
                ps = [p for p in campagne_di(dataset) if (p, False) in accuratezze]
                x = np.arange(len(ps))
                larghezza = 0.26
                for i, (chiave, etichetta, colore) in enumerate(COLORI_LIVELLI):
                    valori = [accuratezze[(p, False)][chiave] for p in ps]
                    barre = ax.bar(x + (i - 1) * larghezza, valori, larghezza,
                                   color=colore, label=etichetta)
                    ax.bar_label(barre, labels=[_it(v, 1) for v in valori],
                                 fontsize=6.2, rotation=90, padding=2)
                ax.set_xticks(x)
                ax.set_xticklabels([_breve(etichetta_modello(p)) for p in ps],
                                   fontsize=8)
                ax.set_title(dataset, fontsize=10)
                ax.set_ylim(60, 97)
                ax.yaxis.grid(True, linestyle="dotted", alpha=0.5)
                ax.set_axisbelow(True)
                for lato in ("top", "right"):
                    ax.spines[lato].set_visible(False)
                if dataset == "BoolQ":
                    ax.axhline(62.2, color="#B00000", linewidth=1,
                               linestyle="dashed")
                    ax.text(len(ps) - 0.5, 62.7, "classe più frequente (62,2)",
                            fontsize=7, color="#B00000", ha="right")
            axes[0][0].set_ylabel("Accuratezza (%)")
            axes[0][0].legend(loc="upper left", ncol=3, frameon=False,
                              fontsize=8, columnspacing=1.0)
            fig.tight_layout()
            salva_tesi(fig, "cap6_accuratezza_tre_livelli.png", extra=True)

        # --- 6.3 heatmap di Pearson ---------------------------------------
        colonne = [p for d in DATASET_TESI for p in campagne_di(d)
                   if (p, False) in pearson]
        if colonne:
            M = np.array([[pearson[(p, False)][chiave] for p in colonne]
                          for chiave, _ in ETICHETTE_HEATMAP])
            fig, ax = plt.subplots(figsize=(9.2, 4.6))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    v = M[r, c]
                    testo = "n/d" if v != v else f"{v:+.3f}".replace(".", ",")
                    ax.text(c, r, testo, ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.33 else "black")
            # riquadro sul |r| massimo di ogni campagna
            for c in range(M.shape[1]):
                finiti = [(abs(M[r, c]), r) for r in range(M.shape[0])
                          if M[r, c] == M[r, c]]
                if finiti:
                    r = max(finiti)[1]
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               fill=False, edgecolor="black",
                                               linewidth=1.6))
            ax.set_xticks(range(len(colonne)))
            ax.set_xticklabels([_breve(etichetta_modello(p)) for p in colonne],
                               fontsize=8)
            for dataset in DATASET_TESI:
                indici = [i for i, p in enumerate(colonne)
                          if nome_dataset(p) == dataset]
                if indici:
                    ax.text(sum(indici) / len(indici), -0.13, dataset,
                            ha="center", va="top", fontsize=9.5,
                            transform=ax.get_xaxis_transform())
            ax.set_yticks(range(len(ETICHETTE_HEATMAP)))
            ax.set_yticklabels([e for _, e in ETICHETTE_HEATMAP], fontsize=8.5)
            for x in range(1, len(colonne)):
                if nome_dataset(colonne[x]) != nome_dataset(colonne[x - 1]):
                    ax.axvline(x - 0.5, color="black", linewidth=1.4)
            ax.set_xticks(np.arange(-0.5, len(colonne), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(ETICHETTE_HEATMAP), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", length=0)
            barra = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            barra.set_label("$r$ di Pearson con la scorrettezza", fontsize=8.5)
            fig.tight_layout()
            salva_tesi(fig, "cap6_pearson_heatmap.png", extra=True)

        # --- 6.4 griglie 2x2 delle curve accuracy-rejection ----------------
        def griglia_rejection(dataset, nome_file, filtrato=False):
            ps = [p for p in campagne_di(dataset) if dati_curve.get((p, filtrato))]
            if len(ps) < 2:
                return
            righe = int(math.ceil(len(ps) / 2))
            fig, axes = plt.subplots(righe, 2, figsize=(9.2, 3.4 * righe),
                                     sharex=True, squeeze=False)
            piatti = axes.ravel()
            for ax, p in zip(piatti, ps):
                d = dati_curve[(p, filtrato)]
                ax.fill_between(FRAZIONI_RIFIUTATE, d["casuale_min"],
                                d["casuale_max"], color="0.55", alpha=0.30,
                                linewidth=0,
                                label="rifiuto casuale ($\\pm$ 1 dev. std.)")
                ax.plot(FRAZIONI_RIFIUTATE, d["casuale"], color="black",
                        linewidth=1.2)
                for etichetta, chiave, colore, stile in DESCRITTORI_PAPER:
                    if chiave in d["curve"]:
                        ax.plot(FRAZIONI_RIFIUTATE, d["curve"][chiave],
                                color=colore, linestyle=stile, linewidth=1.5,
                                label=etichetta)
                ax.set_title(f"{_breve(etichetta_modello(p))}  "
                             f"(accuratezza {_it(100 * d['accuratezza'], 1)} %)",
                             fontsize=9.5)
                ax.set_xlim(0, 0.95)
                ax.grid(linestyle="dotted", alpha=0.6)
                ax.set_axisbelow(True)
            for ax in piatti[len(ps):]:
                ax.set_visible(False)
            maniglie, etichette = piatti[0].get_legend_handles_labels()
            fig.legend(maniglie, etichette, loc="lower center", ncol=6,
                       frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.015))
            fig.supxlabel("Frazione di domande rifiutate, dalla più incerta",
                          fontsize=9, y=0.055)
            fig.supylabel("Accuratezza sulle domande tenute", fontsize=9)
            fig.tight_layout(rect=(0.01, 0.07, 1, 1))
            salva_tesi(fig, nome_file)

        griglia_rejection("BoolQ", "cap6_rejection_boolq.png")
        griglia_rejection("CommonsenseQA", "cap6_rejection_csqa.png")
        # Stesse griglie nel regime filtrato, per il confronto della sezione
        # sull'impatto della selezione delle varianti stabili.
        griglia_rejection("BoolQ", "cap6_rejection_boolq_filtrato.png",
                          filtrato=True)
        griglia_rejection("CommonsenseQA", "cap6_rejection_csqa_filtrato.png",
                          filtrato=True)

        # --- 6.4 guadagno di area sul rifiuto casuale ----------------------
        if any(dati_curve.get((p, False)) for p in PERCORSI):
            fig, axes = plt.subplots(1, len(DATASET_TESI), figsize=(9.2, 3.6),
                                     sharey=True, squeeze=False)
            for ax, dataset in zip(axes[0], DATASET_TESI):
                ps = [p for p in campagne_di(dataset) if dati_curve.get((p, False))]
                x = np.arange(len(ps))
                larghezza = 0.16
                for i, (etichetta, chiave, colore, stile) in enumerate(DESCRITTORI_PAPER):
                    valori = [dati_curve[(p, False)]["aree"][chiave]
                              - dati_curve[(p, False)]["area_casuale"] for p in ps]
                    ax.bar(x + (i - 2) * larghezza, valori, larghezza,
                           color=colore, label=etichetta, edgecolor="white",
                           linewidth=0.5,
                           hatch="////" if stile == "dotted" else None)
                ax.set_xticks(x)
                ax.set_xticklabels([_breve(etichetta_modello(p)) for p in ps],
                                   fontsize=8)
                ax.set_title(dataset, fontsize=10)
                ax.yaxis.grid(True, linestyle="dotted", alpha=0.5)
                ax.set_axisbelow(True)
                for lato in ("top", "right"):
                    ax.spines[lato].set_visible(False)
            axes[0][0].set_ylabel("Guadagno di area sul rifiuto casuale")
            axes[0][0].legend(loc="upper left", ncol=5, frameon=False,
                              fontsize=8, columnspacing=0.9, handlelength=1.4)
            fig.tight_layout()
            salva_tesi(fig, "cap6_aree_descrittori.png", extra=True)

        # --- 6.5 scomposizione AU/EU fra modelli ---------------------------
        if au_eu_modelli:
            fig, ax = plt.subplots(figsize=(6.4, 3.9))
            etichette, valori_au, valori_eu, posizioni = [], [], [], []
            posizione = 0.0
            for dataset in DATASET_TESI:
                if dataset not in au_eu_modelli:
                    continue
                au_o, eu_o, au_m, eu_m = au_eu_modelli[dataset][:4]
                for etichetta, au, eu in (("Domanda\noriginale", au_o, eu_o),
                                          ("Media delle\nvarianti", au_m, eu_m)):
                    etichette.append(etichetta)
                    valori_au.append(au)
                    valori_eu.append(eu)
                    posizioni.append(posizione)
                    posizione += 1
                posizione += 0.6
            ax.bar(posizioni, valori_au, width=0.7, color="#26456E", label="AU")
            ax.bar(posizioni, valori_eu, width=0.7, bottom=valori_au,
                   color="#A03030", label="EU")
            for x, au, eu in zip(posizioni, valori_au, valori_eu):
                ax.text(x, au / 2, _it(au, 3), ha="center", va="center",
                        fontsize=8, color="white")
                ax.text(x, au + eu / 2, _it(eu, 3), ha="center", va="center",
                        fontsize=8, color="white")
                ax.text(x, au + eu + 0.006, _it(au + eu, 3), ha="center",
                        va="bottom", fontsize=8, color="#333333")
            ax.set_xticks(posizioni)
            ax.set_xticklabels(etichette, fontsize=8)
            ax.set_ylim(0, max(a + e for a, e in zip(valori_au, valori_eu)) * 1.28)
            ax.set_ylabel("Incertezza (entropia normalizzata)")
            ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5)
            ax.yaxis.grid(True, linestyle="dotted", alpha=0.5)
            ax.set_axisbelow(True)
            for lato in ("top", "right"):
                ax.spines[lato].set_visible(False)
            passo = 0
            for dataset in DATASET_TESI:
                if dataset in au_eu_modelli:
                    centro = (posizioni[passo] + posizioni[passo + 1]) / 2
                    ax.text(centro, -0.17, dataset, ha="center", va="top",
                            fontsize=9.5, transform=ax.get_xaxis_transform())
                    passo += 2
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            salva_tesi(fig, "cap6_au_eu_modelli.png")

        # --- 6.6 la coppia a divergenza massima ----------------------------
        p_esempio = campagna_esempio("BoolQ")
        voce = coppie_max_kl.get((p_esempio, False)) if p_esempio else None
        if voce:
            kl, riga, i, j, M = voce
            classi = classi_per_dataset[nome_dataset(p_esempio)]
            ents = entropie(M)
            fig, ax = plt.subplots(figsize=(6.6, 3.1))
            x = np.arange(len(classi))
            for k, (indice, colore) in enumerate(((i, "#3C6E9F"), (j, "#D08A45"))):
                barre = ax.bar(x + (k - 0.5) * 0.36, M[indice], 0.36,
                               color=colore,
                               label=f"formulazione {'AB'[k]}"
                                     f"   H = {_it(float(ents[indice]), 3)}")
                ax.bar_label(barre, labels=[_it(v, 3) for v in M[indice]],
                             fontsize=7.5, padding=2)
            ax.set_xticks(x)
            ax.set_xticklabels(list(classi), fontsize=9, style="italic")
            ax.set_ylim(0, 1.18)
            ax.set_ylabel("Probabilità assegnata")
            ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2)
            ax.yaxis.grid(True, linestyle="dotted", alpha=0.5)
            ax.set_axisbelow(True)
            for lato in ("top", "right"):
                ax.spines[lato].set_visible(False)
            fig.tight_layout()
            salva_tesi(fig, "cap6_coppia_kl_massima.png", extra=True)
            print(f"    coppia KL massima usata: {etichetta_modello(p_esempio)}"
                  f" - {riga['domanda']} - KLsym = {kl:.4f}")

        # --- 6.7 effetto del filtro di convergenza -------------------------
        p_filtro = campagna_esempio("CommonsenseQA")
        if p_filtro and dati_curve.get((p_filtro, False)) and dati_curve.get((p_filtro, True)):
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
            for ax, filtrato in zip(axes, (False, True)):
                d = dati_curve[(p_filtro, filtrato)]
                ax.fill_between(FRAZIONI_RIFIUTATE, d["casuale_min"],
                                d["casuale_max"], color="0.55", alpha=0.30,
                                linewidth=0,
                                label="rifiuto casuale ($\\pm$ 1 dev. std.)")
                ax.plot(FRAZIONI_RIFIUTATE, d["casuale"], color="black",
                        linewidth=1.2)
                for etichetta, chiave, colore, stile in DESCRITTORI_PAPER:
                    if chiave in d["curve"]:
                        ax.plot(FRAZIONI_RIFIUTATE, d["curve"][chiave],
                                color=colore, linestyle=stile, linewidth=1.5,
                                label=etichetta)
                titolo = ("con selezione delle varianti stabili" if filtrato
                          else "tutte le ripetizioni")
                ax.set_title(f"{titolo}\n{d['domande']} domande, accuratezza "
                             f"{_it(100 * d['accuratezza'], 1)} %", fontsize=9.5)
                ax.set_xlim(0, 0.95)
                ax.grid(linestyle="dotted", alpha=0.6)
                ax.set_axisbelow(True)
            maniglie, etichette = axes[0].get_legend_handles_labels()
            fig.legend(maniglie, etichette, loc="lower center", ncol=6,
                       frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
            fig.supxlabel("Frazione di domande rifiutate, dalla più incerta",
                          fontsize=9, y=0.08)
            axes[0].set_ylabel("Accuratezza sulle domande tenute")
            fig.tight_layout(rect=(0, 0.10, 1, 1))
            salva_tesi(fig, "cap6_filtro_prima_dopo.png", extra=True)
            print(f"    effetto del filtro mostrato su "
                  f"{etichetta_modello(p_filtro)} - {nome_dataset(p_filtro)}")


print("\nFatto.")
