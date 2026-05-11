import { type FormEvent, type ReactNode, useMemo, useState } from 'react';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

type Verticale = 'relocation' | 'life_on_campus' | 'study_abroad' | 'career_readiness';
type VerticalFilter = Verticale | 'all';

type AskResponse = {
  answer: string;
  sources: string[];
  verticale: Verticale;
};

type ChatItem = {
  id: number;
  question: string;
  response?: AskResponse;
  latencyMs?: number;
  error?: string;
};

type IconName =
  | 'arrow'
  | 'arrowUpRight'
  | 'bot'
  | 'briefcase'
  | 'file'
  | 'globe'
  | 'home'
  | 'map'
  | 'plus'
  | 'sparkles'
  | 'user'
  | 'x';

const verticali: Array<{
  id: VerticalFilter;
  label: string;
  eyebrow: string;
  prompt?: string;
  icon: IconName;
}> = [
  {
    id: 'all',
    label: 'All',
    eyebrow: 'AUTO',
    icon: 'sparkles',
  },
  {
    id: 'relocation',
    label: 'Moving to Milan',
    eyebrow: 'ARRIVE',
    prompt:
      "List the steps an international student must follow to register with Italy's National Health Service in Milan.",
    icon: 'map',
  },
  {
    id: 'life_on_campus',
    label: 'Life on campus',
    eyebrow: 'LIVE',
    prompt: 'Provide a structured table of the dining areas available on the Bocconi campus.',
    icon: 'home',
  },
  {
    id: 'study_abroad',
    label: 'Exchange & abroad',
    eyebrow: 'GO GLOBAL',
    prompt:
      'How are GPA, credits, and Bachelor degree grade weighted for the MSc Exchange Program selection score?',
    icon: 'globe',
  },
  {
    id: 'career_readiness',
    label: 'Career & salaries',
    eyebrow: 'NEXT STEP',
    prompt: 'What is the maximum amount of the Bocconi Merit Award tuition waiver for graduate students?',
    icon: 'briefcase',
  },
];

const suggestedPrompts = [
  'What dining options are available on campus?',
  'Which universities can I exchange with from BIEM in Asia?',
  'How do I apply for the Bocconi Graduate Merit Award?',
  'How much cheaper is the ATM annual pass for students under 27?',
];

const metrics = [
  { value: '6h', label: 'hackathon sprint' },
  { value: '1,617', label: 'source files' },
  { value: '6,433', label: 'indexed chunks' },
  { value: '<5s', label: 'typical reply' },
  { value: '100%', label: 'grounded mode' },
];

function formatVerticale(value: Verticale) {
  return value.replaceAll('_', ' ');
}

function iconForVerticale(value: Verticale): IconName {
  if (value === 'relocation') return 'map';
  if (value === 'life_on_campus') return 'home';
  if (value === 'study_abroad') return 'globe';
  return 'briefcase';
}

function shortSource(path: string) {
  const parts = path.split('/');
  return parts[parts.length - 1] || path;
}

function Icon({ name, className = '' }: { name: IconName; className?: string }) {
  const common = {
    className: `icon ${className}`.trim(),
    fill: 'none',
    stroke: 'currentColor',
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    strokeWidth: 1.8,
    viewBox: '0 0 24 24',
  };

  const paths: Record<IconName, ReactNode> = {
    arrow: (
      <>
        <path d="M12 19V5" />
        <path d="m5 12 7-7 7 7" />
      </>
    ),
    arrowUpRight: (
      <>
        <path d="M7 17 17 7" />
        <path d="M9 7h8v8" />
      </>
    ),
    bot: (
      <>
        <path d="M12 8V5" />
        <rect height="11" rx="4" width="14" x="5" y="8" />
        <path d="M9 13h.01" />
        <path d="M15 13h.01" />
        <path d="M9 17h6" />
      </>
    ),
    briefcase: (
      <>
        <path d="M10 7V6a2 2 0 0 1 2-2h0a2 2 0 0 1 2 2v1" />
        <rect height="12" rx="3" width="18" x="3" y="7" />
        <path d="M3 12h18" />
      </>
    ),
    file: (
      <>
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
        <path d="M14 3v5h5" />
      </>
    ),
    globe: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M3 12h18" />
        <path d="M12 3a14 14 0 0 1 0 18" />
        <path d="M12 3a14 14 0 0 0 0 18" />
      </>
    ),
    home: (
      <>
        <path d="m4 11 8-7 8 7" />
        <path d="M6 10v9h12v-9" />
        <path d="M10 19v-5h4v5" />
      </>
    ),
    map: (
      <>
        <path d="M12 21s6-5.1 6-11a6 6 0 0 0-12 0c0 5.9 6 11 6 11Z" />
        <circle cx="12" cy="10" r="2.2" />
      </>
    ),
    plus: (
      <>
        <path d="M12 5v14" />
        <path d="M5 12h14" />
      </>
    ),
    sparkles: (
      <>
        <path d="m12 3 1.6 4.2L18 9l-4.4 1.8L12 15l-1.6-4.2L6 9l4.4-1.8Z" />
        <path d="m19 14 .8 2 2.2 1-.2.1-2 1-.8 2-.8-2-2.2-1 2.2-1Z" />
        <path d="m4.5 14 .6 1.4 1.4.6-1.4.6-.6 1.4-.6-1.4-1.4-.6 1.4-.6Z" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
    x: (
      <>
        <path d="M18 6 6 18" />
        <path d="m6 6 12 12" />
      </>
    ),
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [question, setQuestion] = useState('');
  const [chat, setChat] = useState<ChatItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedVertical, setSelectedVertical] = useState<VerticalFilter>('all');
  const [activeSource, setActiveSource] = useState<string | null>(null);

  const hasChat = chat.length > 0;

  const latestSources = useMemo(() => {
    const latest = [...chat].reverse().find((item) => item.response?.sources.length);
    return latest?.response?.sources ?? [];
  }, [chat]);

  async function submitQuestion(nextQuestion = question) {
    const trimmed = nextQuestion.trim();
    if (!trimmed || isLoading) return;

    const id = Date.now();
    const startedAt = performance.now();
    setChat((items) => [...items, { id, question: trimmed }]);
    setQuestion('');
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      });
      const data = (await response.json()) as AskResponse;
      const latencyMs = Math.round(performance.now() - startedAt);
      setChat((items) =>
        items.map((item) => (item.id === id ? { ...item, response: data, latencyMs } : item)),
      );
    } catch {
      setChat((items) =>
        items.map((item) =>
          item.id === id
            ? { ...item, error: 'I could not reach the backend. Please try again.' }
            : item,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion();
  }

  function startNewChat() {
    setChat([]);
    setQuestion('');
    setSelectedVertical('all');
    setActiveSource(null);
  }

  return (
    <main className={`app-shell ${hasChat ? 'has-chat' : ''}`}>
      <div className="aurora" aria-hidden="true" />

      <header className="topbar">
        <a className="brand" href="/" aria-label="Bocconi Buddy home">
          <span className="logo-mark">
            <svg
              aria-hidden="true"
              fill="none"
              viewBox="0 0 32 32"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M9 6.5V25.5"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="2"
              />
              <path
                d="M9 7H17.2C20.4 7 22.5 8.85 22.5 11.55C22.5 14.25 20.4 16 17.2 16H9"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="2"
              />
              <path
                d="M9 16H18.4C21.75 16 24 17.85 24 20.55C24 23.25 21.75 25 18.4 25H9"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="2"
              />
              <path
                d="M12.4 22.7C15.3 18.6 17.1 14.2 18.8 9.2"
                stroke="url(#buddyMarkGradient)"
                strokeLinecap="round"
                strokeWidth="1.6"
              />
              <path
                d="M20.2 8.9L19.1 8.9L19.45 7.85"
                stroke="url(#buddyMarkGradient)"
                strokeLinecap="round"
                strokeWidth="1.2"
              />
              <defs>
                <linearGradient
                  gradientUnits="userSpaceOnUse"
                  id="buddyMarkGradient"
                  x1="12.4"
                  x2="20.2"
                  y1="22.7"
                  y2="8.2"
                >
                  <stop stopColor="#8B5CF6" />
                  <stop offset="1" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
          </span>
          <span className="brand-wordmark">
            Bocconi <em>Buddy</em>
          </span>
        </a>

        <div className="live-pill" aria-label="Backend online">
          <span className="live-dot" />
          <span>ONLINE</span>
        </div>
      </header>

      <section className="hero-chat" aria-label="Ask Bocconi Buddy">
        <div className="hero-copy">
          <p className="eyebrow">BOCCONI · STUDENT ASSISTANT</p>
          <h1>
            What do you <span>want</span>
            <br />
            to figure out today?
          </h1>
          <p className="subhead">
            Honest answers from real Bocconi sources. Built by a student, for students.
          </p>
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="question">
            Ask me anything
          </label>
          <Icon name="sparkles" className="composer-spark" />
          <input
            autoComplete="off"
            id="question"
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={hasChat ? 'Ask a follow-up...' : 'Ask me anything — in English or Italian'}
            value={question}
          />
          {hasChat && (
            <button
              aria-label="Start a new chat"
              className="new-chat-button"
              onClick={startNewChat}
              type="button"
            >
              <Icon name="plus" />
            </button>
          )}
          <button
            aria-label="Send question"
            className="send-button"
            disabled={isLoading || !question.trim()}
            type="submit"
          >
            <Icon name="arrow" />
          </button>
        </form>

        <div className="vertical-rail" aria-label="Question categories">
          {verticali.map((item) => (
            <button
              aria-pressed={selectedVertical === item.id}
              className={`vertical-chip ${item.id} ${selectedVertical === item.id ? 'active' : ''}`}
              key={item.id}
              onClick={() => setSelectedVertical(item.id)}
              type="button"
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </div>

        {!hasChat ? (
          <section className="suggestions" aria-label="Suggested prompts">
            <p className="section-label">TRY ONE OF THESE</p>
            <div className="prompt-grid">
              {suggestedPrompts.map((prompt) => (
                <button
                  className="prompt-card"
                  key={prompt}
                  onClick={() => void submitQuestion(prompt)}
                  type="button"
                >
                  <span>{prompt}</span>
                  <Icon name="arrowUpRight" />
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="thread" aria-live="polite">
            {chat.map((item) => (
              <article className="exchange" key={item.id}>
                <div className="message-row user-row">
                  <span className="avatar user-avatar">
                    <Icon name="user" />
                  </span>
                  <p className="user-message">{item.question}</p>
                </div>

                <div className="response-card">
                  {item.response ? (
                    <>
                      <div className="response-topline">
                        <span className="mini-logo">
                          <Icon name="sparkles" />
                        </span>
                        <span className={`vertical-label ${item.response.verticale}`}>
                          <Icon name={iconForVerticale(item.response.verticale)} />
                          {formatVerticale(item.response.verticale)}
                        </span>
                        <span className="latency">⚡ {(item.latencyMs ?? 0) / 1000}s</span>
                      </div>

                      <div className="answer">
                        {item.response.answer.split('\n').map((line, index) => (
                          <p key={`${item.id}-${index}`}>{line}</p>
                        ))}
                      </div>

                      {item.response.sources.length > 0 && (
                        <div className="sources">
                          <div className="source-separator" />
                          <p className="source-label">SOURCES</p>
                          <div className="source-chips">
                            {item.response.sources.map((source) => (
                              <button
                                className="source-chip"
                                key={source}
                                onClick={() => setActiveSource(source)}
                                type="button"
                              >
                                <Icon name="file" />
                                <span>{shortSource(source)}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : item.error ? (
                    <p className="error-text">{item.error}</p>
                  ) : (
                    <div className="thinking">
                      <span className="thinking-dot" />
                      Reading the sources...
                    </div>
                  )}
                </div>
              </article>
            ))}
          </section>
        )}
      </section>

      <section className="stats-strip" aria-label="Project stats">
        {metrics.map((metric) => (
          <div className="metric" key={metric.label}>
            <strong>{metric.value}</strong>
            <span>{metric.label}</span>
          </div>
        ))}
      </section>

      <footer className="footer">
        <div>
          <h2>What is Buddy?</h2>
          <p>
            A practical student assistant for Bocconi life, from settling into Milan to checking
            exchange rules and career facts. It is designed to answer when the sources support it
            and step back when they do not.
          </p>
        </div>
        <div>
          <h2>How it works</h2>
          <ul>
            <li>Hybrid retrieval over Bocconi and public sources</li>
            <li>6,433 indexed chunks across four student verticals</li>
            <li>Grounding rules that prefer silence over guessing</li>
          </ul>
        </div>
        <div>
          <h2>Built by David</h2>
          <p>BIEM student · 2026 Bocconi AI Hackathon</p>
          <div className="footer-links">
            <a href="https://www.linkedin.com" rel="noreferrer" target="_blank">
              LinkedIn
            </a>
            <a href="https://github.com" rel="noreferrer" target="_blank">
              GitHub
            </a>
          </div>
        </div>
      </footer>

      {activeSource && (
        <div
          className="modal-backdrop"
          onClick={() => setActiveSource(null)}
          role="presentation"
        >
          <dialog
            className="source-modal"
            onClick={(event) => event.stopPropagation()}
            open
          >
            <button
              aria-label="Close source preview"
              className="modal-close"
              onClick={() => setActiveSource(null)}
              type="button"
            >
              <Icon name="x" />
            </button>
            <p className="section-label">SOURCE PATH</p>
            <h2>{shortSource(activeSource)}</h2>
            <p className="modal-path">{activeSource}</p>
            <p className="modal-note">
              This is the exact source path returned by the grounded answer. Keep it attached when
              you want to verify or inspect the supporting document.
            </p>
          </dialog>
        </div>
      )}
    </main>
  );
}

export default App;
