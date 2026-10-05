import { useEffect, useState } from 'react';

const emotionLabels = ['happy', 'sad', 'angry', 'relaxed'];

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [signalInput, setSignalInput] = useState('0.1,0.2,0.3,0.4,0.5');

  useEffect(() => {
    fetch('/model-info')
      .then((response) => response.json())
      .then((data) => setModelInfo(data))
      .catch(() => setModelInfo({ status: 'not_trained' }));
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    const signal = signalInput.split(',').map(Number);
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal }),
    });
    const data = await response.json();
    setPrediction(data);
    setLoading(false);
  };

  const handleTrain = async () => {
    setLoading(true);
    const response = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_path: 'Dataset' }),
    });
    const data = await response.json();
    setModelInfo({ status: 'trained', metrics: data.metrics });
    setLoading(false);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>NeuroSense AI</h1>
        <p>Emotion recognition from EEG-style signals</p>
        <nav>
          <a href="#">Home</a>
          <a href="#">Dataset</a>
          <a href="#">Training</a>
          <a href="#">Prediction</a>
          <a href="#">Performance</a>
          <a href="#">Graphs</a>
          <a href="#">About</a>
        </nav>
      </aside>
      <main className="main-content">
        <section className="hero-card">
          <div>
            <h2>Hardware-free EEG emotion recognition</h2>
            <p>Train models on DEAP-inspired features and deploy them through a production-ready API.</p>
          </div>
          <div className="hero-actions">
            <button onClick={handleTrain}>Train Models</button>
            <button onClick={handlePredict}>Predict Emotion</button>
          </div>
        </section>

        <section className="stats-grid">
          <div className="card">
            <h3>Model Status</h3>
            <p>{modelInfo?.status || 'loading...'}</p>
          </div>
          <div className="card">
            <h3>Best Accuracy</h3>
            <p>{modelInfo?.metrics?.random_forest?.accuracy?.toFixed(2) || '—'}</p>
          </div>
          <div className="card">
            <h3>Prediction</h3>
            <p>{prediction?.prediction || '—'}</p>
          </div>
          <div className="card">
            <h3>Confidence</h3>
            <p>{prediction?.confidence?.toFixed(2) || '—'}</p>
          </div>
        </section>

        <section className="card">
          <h3>Prediction Input</h3>
          <textarea value={signalInput} onChange={(event) => setSignalInput(event.target.value)} rows={5} />
          {loading ? <p>Processing...</p> : null}
          {prediction ? (
            <div>
              <h4>Probabilities</h4>
              <ul>
                {emotionLabels.map((label) => (
                  <li key={label}>{label}: {(prediction.probabilities?.[label] * 100).toFixed(1)}%</li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      </main>
    </div>
  );
}

export default App;
