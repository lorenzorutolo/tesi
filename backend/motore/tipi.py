"""Strutture dati e contratto delle specifiche di dataset.

Il motore lavora su queste strutture senza sapere nulla del dataset concreto:
ogni dataset si descrive implementando il Protocol ``DatasetSpec``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Domanda:
    """Una domanda da porre al modello, in una sua specifica formulazione.

    ``testo`` e ``dati`` possono cambiare tra una variante e l'altra (parafrasi,
    shuffle delle opzioni), mentre la spec del dataset sa come renderli in prompt.
    """
    testo: str                                  # stem da porre (puo' essere parafrasato)
    contesto: str = ""                          # passage per BoolQ; "" se assente
    reale: str = ""                             # classe gold per QUESTA formulazione
    dati: dict = field(default_factory=dict)    # spec-specifico (es. opzioni/ordine MC)


@dataclass
class Alternativa:
    """Esito di una singola interrogazione del modello.

    ``probabilita`` e' una distribuzione sulle classi dichiarate dal dataset, con
    in piu' la chiave "altro" sempre presente. Esempi:
      - BoolQ: {"true": .., "false": .., "altro": ..}
      - MC 4 opzioni: {"A": .., "B": .., "C": .., "D": .., "altro": ..}

    ``ordine`` registra, per i dataset che mostrano opzioni (MC), la sequenza
    delle lettere canoniche nell'ordine in cui sono state presentate al modello
    (es. ["C","A","E","B","D"]): senza, non si potrebbe ricostruire quale
    permutazione ha prodotto questa risposta. None per i dataset senza opzioni.
    """
    domanda_alt: str
    risposta_pulita: str            # classe vincente (argmax), es. "true" o "C"
    probabilita: dict[str, float]   # {classe: prob_raw}; "altro" sempre presente
    convergente: bool = True
    ordine: list[str] | None = None # lettere nell'ordine mostrato (solo MC)


@dataclass
class RigaBenchmark:
    id: int
    domanda: str
    reale: str
    alternative: list[Alternativa] = field(default_factory=list)
    scartate: list[Alternativa] = field(default_factory=list)


class DatasetSpec(Protocol):
    """Contratto che ogni dataset deve implementare per essere eseguito dal motore.

    Tutto cio' che e' specifico del dataset vive qui dentro; il motore resta
    agnostico. Esempi di implementazioni:
    - BoolQ: ``classi`` = ["true","false"], perturbazione = sola parafrasi.
    - Multiple-choice: ``classi`` sono le lettere delle
      opzioni (es. ["A","B","C","D","E"]); la perturbazione e' lo shuffle delle
      opzioni (salvate in ``Domanda.dati``). Le lettere sono INCOLLATE al
      contenuto: ogni opzione conserva la propria lettera anche dopo lo shuffle,
      che cambia solo l'ORDINE di presentazione. Cosi' la lettera scelta dal
      modello e' gia' la chiave canonica e le distribuzioni restano stabili e
      confrontabili/mediabili tra le varianti, senza alcun rimappaggio.
    """

    nome: str           # identificatore usato dalla CLI (es. "boolq")
    hf_id: str          # id del dataset su HuggingFace (es. "google/boolq")
    split: str          # split da usare (es. "validation")
    classi: list[str]   # classi di risposta dichiarate (es. ["true", "false"])

    def leggi_riga(self, raw: dict) -> Domanda:
        """Estrae una Domanda dalla riga grezza del dataset."""
        ...

    def prompt_risposta(self, d: Domanda) -> str:
        """Costruisce il prompt con cui interrogare il modello su ``d``."""
        ...

    def classe_di_token(self, token: str, d: Domanda) -> str | None:
        """Mappa un token del modello a una classe CANONICA, o None se non
        riconosciuto (in tal caso la probabilita' finisce nel bucket "altro").

        Riceve la ``Domanda`` corrente per i dataset che ne avessero bisogno, ma
        le implementazioni attuali non la usano: nel multiple-choice le lettere
        sono incollate al contenuto (la lettera mostrata e' gia' canonica) e in
        BoolQ non c'e' rimescolamento, quindi ``d`` viene ignorato."""
        ...

    def genera_variante(self, corrente: Domanda,
                        parafrasa: Callable[[str], str]) -> Domanda:
        """Produce la prossima variante a partire da ``corrente``.

        ``parafrasa`` e' iniettato dal motore: e' un servizio che dato un testo
        ne restituisce una parafrasi tramite il modello. La spec decide come
        usarlo (solo parafrasi, parafrasi + shuffle opzioni, ecc.)."""
        ...

    # Hook OPZIONALI (il motore li cerca via getattr, come ``opzioni_mostrate``):
    #
    # def varianti_esaustive(self, d: Domanda) -> Iterator[Domanda]:
    #     """Enumera TUTTE le varianti possibili di ``d``, senza ripetizioni,
    #     con l'originale per prima. Ha senso solo quando lo spazio delle
    #     perturbazioni e' finito (es. MC: le K! permutazioni delle opzioni);
    #     BoolQ non lo definisce (le parafrasi non sono enumerabili). Usato
    #     dalla modalita' esaustiva del benchmark, che interroga ogni variante
    #     una volta sola, senza loop di convergenza ne' scarti."""
