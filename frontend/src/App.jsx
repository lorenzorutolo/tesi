import { useState } from 'react'

export default function App() {
  const [indice, setIndice] = useState('')
  const [risultato, setRisultato] = useState(null)
  const [inCaricamento, setInCaricamento] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (indice === '') return

    setInCaricamento(true)
    setRisultato(null)

    // TODO: chiamata al backend, per ora segnaposto
    setRisultato({
      risposta: null,
      entropia: null,
      probabilita: null,
    })
    setInCaricamento(false)
  }

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: '720px' }}>
      <h1>Interfaccia Web</h1>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '2rem' }}>
        <label htmlFor="indice">Indice domanda:</label>
        <input
          id="indice"
          type="number"
          min="0"
          value={indice}
          onChange={(e) => setIndice(e.target.value)}
          required
          style={{ padding: '0.4rem', width: '120px' }}
        />
        <button type="submit" disabled={inCaricamento} style={{ padding: '0.4rem 1rem' }}>
          {inCaricamento ? 'Interrogazione...' : 'Interroga modello'}
        </button>
      </form>

      <section>
        <h2>Risultato</h2>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Risposta</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.risposta ?? <em style={{ color: '#888' }}>nessuna risposta</em>}
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Entropia</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.entropia ?? <em style={{ color: '#888' }}>nessun valore</em>}
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <h3>Probabilità</h3>
          <div style={{ padding: '0.75rem', border: '1px solid #ccc', borderRadius: '4px', minHeight: '2rem' }}>
            {risultato?.probabilita ?? <em style={{ color: '#888' }}>nessun valore</em>}
          </div>
        </div>
      </section>
    </main>
  )
}
