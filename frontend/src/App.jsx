import { useState } from 'react'

const API_URL = '/api/interroga'

// Limite lato client, allineato a MAX_RIPETIZIONI_LIVE del backend.
const MAX_RIPETIZIONI = 30
// L'insieme credale ha senso solo con almeno due distribuzioni da confrontare.
const MIN_RIPETIZIONI_CREDALI = 2

// Le due letture dell'incertezza offerte dall'interfaccia.
//   bayesiana: una sola interrogazione, la confidenza e' 1 - H della
//              distribuzione predittiva (nessuna ripetizione necessaria: la
//              probabilita' viene dai logprob, non dalla dispersione fra
//              riformulazioni);
//   credale:   N riformulazioni della stessa domanda formano l'insieme credale
//              e la confidenza e' 1 - AU_C, dove AU_C = max H[P] e' la lettura
//              worst-case dell'incertezza aleatoria.
const BAYESIANA = 'bayesian'
const CREDALE = 'credal'

// Distribuzione normalizzata di UNA ripetizione, come vettore [true, false, altro].
// La normalizzazione per ripetizione serve perche' la massa grezza top-K non e'
// mai esattamente 1: senza, le ripetizioni peserebbero in modo diverso.
function distribuzione(alternativa) {
  const p = alternativa.probabilita || {}
  const v = [p.true || 0, p.false || 0, p.altro || 0]
  const massa = v[0] + v[1] + v[2]
  return massa > 0 ? v.map((x) => x / massa) : null
}

// L'insieme delle distribuzioni valide restituite dall'endpoint: e' l'insieme
// credale della domanda quando le ripetizioni sono piu' di una.
function insiemeDistribuzioni(alternative) {
  if (!alternative || alternative.length === 0) return []
  return alternative.map(distribuzione).filter(Boolean)
}

// Centro di massa dell'insieme (media aritmetica, distribuzione di secondo
// ordine uniforme). Con una sola ripetizione coincide con la distribuzione stessa.
function centroDiMassa(distribuzioni) {
  if (distribuzioni.length === 0) return null
  const somma = [0, 0, 0]
  for (const d of distribuzioni) {
    somma[0] += d[0]
    somma[1] += d[1]
    somma[2] += d[2]
  }
  return somma.map((x) => x / distribuzioni.length)
}

// Entropia ternaria di Shannon in base 3, così il massimo (esiti equiprobabili) è 1.
// La terza classe e' "altro": e' il regime "con altro" usato nell'analisi.
function entropia(d) {
  const log3 = Math.log(3)
  let h = 0
  for (const p of d) {
    if (p > 0) h -= (p * Math.log(p)) / log3
  }
  return h
}

// Risposta secondo il criterio "maggioranza": argmax sul centro di massa.
// Le etichette sono quelle mostrate a schermo, quindi in inglese.
function calcolaRisposta(pmf) {
  if (!pmf) return null
  const [pTrue, pFalse, pAltri] = pmf
  if (pTrue >= pFalse && pTrue >= pAltri) return 'true'
  if (pFalse >= pAltri) return 'false'
  return 'other'
}

// Confidenza mostrata all'utente, secondo la modalita' scelta:
//   bayesiana: 1 - H[P] della singola distribuzione predittiva;
//   credale:   1 - AU_C, con AU_C = massima entropia nell'insieme credale.
function calcolaConfidenza(distribuzioni, modalita) {
  if (distribuzioni.length === 0) return null
  if (modalita === CREDALE) {
    return 1 - Math.max(...distribuzioni.map(entropia))
  }
  return 1 - entropia(distribuzioni[0])
}

// Semaforo a soglie fisse sulla confidenza (sono le vecchie soglie di entropia
// 0.33 / 0.66 lette al contrario): verde sopra 0.67, giallo tra 0.34 e 0.67,
// rosso sotto — segnala a colpo d'occhio quando il modello è "confuso".
const CONFIDENZA_ALTA = 0.67
const CONFIDENZA_MEDIA = 0.34

function coloreConfidenza(confidenza) {
  if (confidenza >= CONFIDENZA_ALTA) return '#2e9e44'
  if (confidenza >= CONFIDENZA_MEDIA) return '#e0b000'
  return '#d0342c'
}

// Formatta la confidenza senza appiattirla sugli estremi: parte da 4 decimali e
// ne aggiunge quanti servono perché un valore vicinissimo a 0 o a 1 non venga
// arrotondato proprio a 0 o a 1.
function formattaConfidenza(confidenza) {
  const distanza = Math.min(Math.abs(confidenza), Math.abs(1 - confidenza))
  const decimali =
    distanza > 0
      ? Math.min(12, Math.max(4, Math.ceil(-Math.log10(distanza)) + 1))
      : 4
  return confidenza.toFixed(decimali)
}

export default function App() {
  const [domanda, setDomanda] = useState('')
  const [modalita, setModalita] = useState(BAYESIANA)
  const [ripetizioni, setRipetizioni] = useState(10)
  const [risultato, setRisultato] = useState(null)
  const [inCaricamento, setInCaricamento] = useState(false)
  const [errore, setErrore] = useState(null)

  // In modalita' bayesiana basta la singola interrogazione: il campo
  // ripetizioni non viene nemmeno mostrato.
  const ripetizioniRichieste =
    modalita === CREDALE ? Math.max(MIN_RIPETIZIONI_CREDALI, ripetizioni) : 1

  const cambiaModalita = (nuova) => {
    setModalita(nuova)
    setRisultato(null)      // la confidenza cambia definizione: non mescolarle
    setErrore(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const testo = domanda.trim()
    if (!testo) return

    setInCaricamento(true)
    setRisultato(null)
    setErrore(null)

    try {
      // Chiede al backend di interrogare il modello dal vivo sulla domanda
      // scritta dall'utente (originale + eventuali parafrasi). Puo' impiegare
      // vari secondi, soprattutto con molte ripetizioni.
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domanda: testo, ripetizioni: ripetizioniRichieste }),
      })
      if (!res.ok) throw new Error(`Server error (HTTP ${res.status})`)
      const riga = await res.json()
      if (riga.errore) throw new Error(riga.errore)

      const distribuzioni = insiemeDistribuzioni(riga.alternative)
      setRisultato({
        risposta: calcolaRisposta(centroDiMassa(distribuzioni)),
        confidenza: calcolaConfidenza(distribuzioni, modalita),
      })
    } catch (err) {
      setErrore(err.message)
    } finally {
      setInCaricamento(false)
    }
  }

  const mostraRisultato = risultato?.risposta != null && risultato?.confidenza != null

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: '720px' }}>
      <h1>LLM with confidence</h1>

      <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center', marginBottom: '1rem' }}>
        <label style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
          <input
            type="radio"
            name="mode"
            value={BAYESIANA}
            checked={modalita === BAYESIANA}
            onChange={() => cambiaModalita(BAYESIANA)}
          />
          Bayesian (single query)
        </label>
        <label style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
          <input
            type="radio"
            name="mode"
            value={CREDALE}
            checked={modalita === CREDALE}
            onChange={() => cambiaModalita(CREDALE)}
          />
          Credal (repeated queries)
        </label>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '2rem' }}>
        <input
          type="text"
          value={domanda}
          onChange={(e) => setDomanda(e.target.value)}
          placeholder="Write a yes/no question (in English)..."
          style={{ flex: 1, padding: '0.4rem 0.6rem' }}
        />
        {modalita === CREDALE && (
          <label style={{ display: 'flex', gap: '0.35rem', alignItems: 'center', whiteSpace: 'nowrap' }}>
            <input
              type="number"
              min={MIN_RIPETIZIONI_CREDALI}
              max={MAX_RIPETIZIONI}
              value={ripetizioni}
              onChange={(e) => setRipetizioni(Number(e.target.value))}
              style={{ width: '4.5rem', padding: '0.4rem' }}
            />
            repetitions
          </label>
        )}
        <button type="submit" disabled={inCaricamento || !domanda.trim()} style={{ padding: '0.4rem 1rem' }}>
          {inCaricamento ? 'Asking...' : 'Ask'}
        </button>
      </form>

      {errore && (
        <div style={{ padding: '0.75rem', border: '1px solid #c33', background: '#fee', color: '#c33', borderRadius: '4px', marginBottom: '1rem' }}>
          Error: {errore}
        </div>
      )}

      <section>
        <h2>Answer</h2>

        <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
          {mostraRisultato ? (
            <span
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: coloreConfidenza(risultato.confidenza) }}
              title={`green >= ${CONFIDENZA_ALTA}, yellow >= ${CONFIDENZA_MEDIA}, red below`}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: '0.9rem',
                  height: '0.9rem',
                  borderRadius: '50%',
                  background: coloreConfidenza(risultato.confidenza),
                }}
              />
              <strong>{risultato.risposta}</strong>
              <span>confidence = {formattaConfidenza(risultato.confidenza)}</span>
            </span>
          ) : (
            <em style={{ color: '#888' }}>no answer yet</em>
          )}
        </div>
      </section>
    </main>
  )
}
