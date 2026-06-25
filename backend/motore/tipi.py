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
    """
    domanda_alt: str
    risposta_pulita: str            # classe vincente (argmax), es. "true" o "C"
    probabilita: dict[str, float]   # {classe: prob_raw}; "altro" sempre presente
    convergente: bool = True


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
      opzioni (salvate in ``Domanda.dati``). La distribuzione e' in chiavi
      CANONICHE: la lettera si riferisce sempre alla posizione ORIGINALE del
      contenuto, non a quella mostrata dopo lo shuffle. Cosi' se il modello
      sceglie sempre lo stesso contenuto la classe vincente resta stabile tra le
      varianti e le distribuzioni sono direttamente confrontabili/mediabili.
      La traduzione lettera-mostrata -> lettera-canonica avviene in
      ``classe_di_token``, che per questo riceve la ``Domanda`` corrente.
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

        Riceve la ``Domanda`` corrente perche' per il multiple-choice la lettera
        mostrata dipende dallo shuffle: "B" sulla variante va tradotto nella
        lettera ORIGINALE dell'opzione corrispondente, cosi' la distribuzione
        resta in chiavi stabili. Per dataset senza rimescolamento (es. BoolQ)
        ``d`` viene ignorato."""
        ...

    def genera_variante(self, corrente: Domanda,
                        parafrasa: Callable[[str], str]) -> Domanda:
        """Produce la prossima variante a partire da ``corrente``.

        ``parafrasa`` e' iniettato dal motore: e' un servizio che dato un testo
        ne restituisce una parafrasi tramite il modello. La spec decide come
        usarlo (solo parafrasi, parafrasi + shuffle opzioni, ecc.)."""
        ...
