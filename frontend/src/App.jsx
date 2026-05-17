import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [healthData, setHealthData] = useState({ status: 'loading', catalog_products: 0 });
  const [messages, setMessages] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(res => res.json())
      .then(data => setHealthData(data))
      .catch(err => {
        console.error("Failed to fetch health:", err);
        setHealthData({ status: 'error', catalog_products: '?' });
      });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const clearConversation = () => {
    setMessages([]);
    setRecommendations([]);
  };

  const handleExampleClick = (example) => {
    setInput(example);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    const turnCount = Math.floor(messages.length / 2) + 1;

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          current_recommendations: recommendations,
          turn_count: turnCount
        })
      });
      
      const data = await response.json();
      
      if (data.recommendations !== null) {
        setRecommendations(data.recommendations);
      }
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        recommendations: data.recommendations
      }]);
      
    } catch (err) {
      console.error("Chat error:", err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* LEFT COLUMN - 30% - CHAT & BRIEF INFO */}
      <div className="left-column">
        <div className="left-header">
          <h2>SHL Assistant</h2>
          <p className="caption">Your AI guide to the SHL assessment catalog.</p>
          <button className="clear-btn" onClick={clearConversation}>Clear Chat</button>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                <div className="chat-message-header">
                  <div className={`chat-avatar ${msg.role}`}>
                    {msg.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                </div>
                
                <div className="chat-content">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  
                  {msg.role === 'assistant' && msg.recommendations && msg.recommendations.length > 0 && (
                    <div className="dataframe-container">
                      <table className="dataframe">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Name</th>
                            <th>Test Type</th>
                            <th>Keys</th>
                            <th>Duration</th>
                            <th>Languages</th>
                            <th>Remote</th>
                            <th>Adaptive</th>
                            <th>URL</th>
                          </tr>
                        </thead>
                        <tbody>
                          {msg.recommendations.map((item, i) => (
                            <tr key={i}>
                              <td>{i + 1}</td>
                              <td>{item.name}</td>
                              <td>{item.test_type}</td>
                              <td>{item.keys.join(', ')}</td>
                              <td>{item.duration}</td>
                              <td>
                                {item.languages.slice(0, 4).join(', ')}
                                {item.languages.length > 4 ? ' +' : ''}
                              </td>
                              <td>{item.remote ? 'True' : 'False'}</td>
                              <td>{item.adaptive ? 'True' : 'False'}</td>
                              <td><a href={item.url} target="_blank" rel="noreferrer">Link</a></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-wrapper">
            <form className="chat-input-form" onSubmit={handleSubmit}>
              <input 
                type="text" 
                className="chat-input"
                placeholder="Describe role, skills, constraints..." 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button type="submit" className="chat-submit" disabled={!input.trim() || isLoading}>
                {isLoading ? <span className="loading-indicator"></span> : 'Send'}
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* RIGHT COLUMN - 70% - DETAILED INFO & ACCESS */}
      <div className="right-column">
        <div className="info-card">
          <h1>What is this App for?</h1>
          <p>
            The SHL Assessment Recommender is an AI-powered conversational agent designed to help hiring managers and recruiters easily navigate the massive SHL product catalog. Instead of manually searching through hundreds of assessments, you can simply describe your hiring needs, required skills, and constraints in plain English. The AI will interactively guide you to the perfect assessment for your candidate.
          </p>

          <div className="status-grid">
            <div className="status-card">
              <span>Catalog Size</span>
              <strong>{healthData.catalog_products} products</strong>
            </div>
            <div className="status-card">
              <span>Retrieval System</span>
              <strong>ChromaDB + Reranker</strong>
            </div>
            <div className="status-card">
              <span>API Endpoints</span>
              <strong>/health and /chat</strong>
            </div>
          </div>
        </div>

        <div className="info-card">
          <h2>How to Access This App</h2>
          <p>
            This platform uses a modern React frontend with a FastAPI backend. You can access the user interface directly through this web application (the window on the left). For developers looking to integrate the recommender into other systems, we also provide a fully-featured REST API.
          </p>
          
          <div className="api-box">
            <strong>Run Local Server:</strong> <code>uvicorn api:app --reload --port 8000</code><br /><br />
            <strong>Interactive API Docs:</strong> Open <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">http://localhost:8000/docs</a> to view the Swagger UI and test the API directly in your browser.
          </div>
        </div>

        <div className="info-card">
          <h2>Try these prompts</h2>
          <p>Click any of the prompts below to automatically fill the chat input and test the recommender:</p>
          <div className="example-row">
            <div className="example-chip" onClick={() => handleExampleClick("Junior AI Developer")}>Junior AI Developer</div>
            <div className="example-chip" onClick={() => handleExampleClick("Senior Java backend engineer")}>Senior Java backend engineer</div>
            <div className="example-chip" onClick={() => handleExampleClick("Graduate management trainee")}>Graduate management trainee</div>
            <div className="example-chip" onClick={() => handleExampleClick("Sales talent audit")}>Sales talent audit</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
