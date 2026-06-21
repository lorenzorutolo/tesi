import { useState } from 'react'

const API_URL = '/api/interroga'

// Aggrega le probabilità grezze su tutte le alternative (originale + ripetizioni)
// e le normalizza in modo che sommino a 1. ``alternative`` arriva gia' come
// array di oggetti dalla risposta JSON dell'endpoint.
function calcolaProbabilita(alternative) {
  if (!alternative || alternative.length === 0) return null

  let sommaTrue = 0
  let sommaFalse = 0
  let sommaAltri = 0
  for (const alt of alternative) {
    const p = alt.probabilita || {}
    sommaTrue += p.true || 0
    sommaFalse += p.false || 0
    sommaAltri += p.altro || 0
  }
  const totale = sommaTrue + sommaFalse + sommaAltri
  if (totale === 0) return null

  return {
    pTrue: sommaTrue / totale,
    pFalse: sommaFalse / totale,
    pAltri: sommaAltri / totale,
    numAlternative: alternative.length,
  }
}

// Risposta secondo il criterio "maggioranza": argmax sulle probabilità aggregate.
function calcolaRisposta(prob) {
  if (!prob) return null
  const { pTrue, pFalse, pAltri } = prob
  if (pTrue >= pFalse && pTrue >= pAltri) return 'true'
  if (pFalse >= pAltri) return 'false'
  return 'altro'
}

// Formatta pTrue/pFalse/pAltri come tre percentuali leggibili con lo stesso numero
// di decimali, scelto in base al più piccolo valore non nullo così che la somma
// visibile sia ~100% e nessuna probabilità venga troncata a "0.00%".
function formattaProbabilita(prob) {
  const pcts = [prob.pTrue * 100, prob.pFalse * 100, prob.pAltri * 100]
  const positivi = pcts.filter(p => p > 0)
  const minNonZero = positivi.length > 0 ? Math.min(...positivi) : 1
  const decimali =
    minNonZero >= 0.01
      ? 2
      : Math.min(8, Math.ceil(-Math.log10(minNonZero)) + 1)
  return {
    pTrue: `${pcts[0].toFixed(decimali)}%`,
    pFalse: `${pcts[1].toFixed(decimali)}%`,
    pAltri: `${pcts[2].toFixed(decimali)}%`,
  }
}

// Entropia ternaria di Shannon in base 3, così il massimo (esiti equiprobabili) è 1.
function calcolaEntropia(prob) {
  if (!prob) return null
  const log3 = Math.log(3)
  let h = 0
  for (const p of [prob.pTrue, prob.pFalse, prob.pAltri]) {
    if (p > 0) h -= (p * Math.log(p)) / log3
  }
  return h
}

export default function App() {
  const [risultato, setRisultato] = useState(null)
  const [inCaricamento, setInCaricamento] = useState(false)
  const [errore, setErrore] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()

    setInCaricamento(true)
    setRisultato(null)
    setErrore(null)

    try {
      // Chiede al backend di estrarre una domanda casuale e interrogare il
      // modello dal vivo (originale + ripetizioni). Puo' impiegare vari secondi.
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: 'boolq' }),
      })
      if (!res.ok) throw new Error(`Errore dal server (HTTP ${res.status})`)
      const riga = await res.json()
      if (riga.errore) throw new Error(riga.errore)

      const probabilita = calcolaProbabilita(riga.alternative)
      setRisultato({
        testoDomanda: riga.domanda,
        risposta: calcolaRisposta(probabilita),
        entropia: calcolaEntropia(probabilita),
        probabilita,
      })
    } catch (err) {
      setErrore(err.message)
    } finally {
      setInCaricamento(false)
    }
  }

  const probFormattate = risultato?.probabilita ? formattaProbabilita(risultato.probabilita) : null

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: '720px' }}>
      <h1>Interfaccia Web</h1>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '2rem' }}>
        <button type="submit" disabled={inCaricamento} style={{ padding: '0.4rem 1rem' }}>
          {inCaricamento ? 'Interrogazione...' : 'Estrai domanda casuale'}
        </button>
      </form>

      {errore && (
        <div style={{ padding: '0.75rem', border: '1px solid #c33', background: '#fee', color: '#c33', borderRadius: '4px', marginBottom: '1rem' }}>
          Errore: {errore}
        </div>
      )}

      <section>
        <h2>Risultato</h2>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Testo domanda</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.testoDomanda ?? <em style={{ color: '#888' }}>nessuna domanda</em>}
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Risposta</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.risposta ?? <em style={{ color: '#888' }}>nessuna risposta</em>}
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Entropia (ternaria, log base 3 · range [0, 1])</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.entropia != null ? risultato.entropia.toFixed(4) : <em style={{ color: '#888' }}>nessun valore</em>}
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Probabilità (aggregata su {risultato?.probabilita?.numAlternative ?? 'N'} ripetizioni)</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.probabilita ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                <div>True: <strong>{probFormattate.pTrue}</strong></div>
                <div>False: <strong>{probFormattate.pFalse}</strong></div>
                <div>Other: <strong>{probFormattate.pAltri}</strong></div>
              </div>
            ) : (
              <em style={{ color: '#888' }}>nessun valore</em>
            )}
          </div>
        </div>
      </section>
    </main>
  )
}
