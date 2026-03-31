import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";

// --- Noise Canvas Component ---
const NoiseCanvas = ({ opacity }) => {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      const imageData = ctx.createImageData(w, h);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const v = Math.random() * 255;
        data[i] = v;
        data[i + 1] = v;
        data[i + 2] = v;
        data[i + 3] = 40;
      }
      ctx.putImageData(imageData, 0, 0);
      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 9999,
        opacity: opacity,
        transition: "opacity 0.3s ease",
      }}
    />
  );
};

// --- Mini Loss Curve (HUD) ---
const MiniLossCurve = ({ progress }) => {
  const w = 120;
  const h = 40;
  const points = [];
  for (let i = 0; i <= 50; i++) {
    const t = i / 50;
    if (t > progress) break;
    const loss = Math.exp(-3.5 * t) + Math.sin(t * 20) * 0.03 * (1 - t);
    const x = (t / 1) * w;
    const y = h - loss * h * 0.9 - 2;
    points.push(`${x},${y}`);
  }
  const currentLoss = Math.exp(-3.5 * progress) + 0.01;
  const epoch = Math.floor(progress * 100);

  return (
    <div
      style={{
        position: "fixed",
        top: 70,
        right: 20,
        zIndex: 10000,
        background: "rgba(0,0,0,0.85)",
        border: "1px solid #2a2a2a",
        borderRadius: 6,
        padding: "10px 14px",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 10,
        color: "#6b6b6b",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ display: "flex", gap: 16, marginBottom: 6 }}>
        <span>
          epoch <span style={{ color: "#e2b55a" }}>{epoch}</span>
        </span>
        <span>
          loss{" "}
          <span style={{ color: "#4ade80" }}>{currentLoss.toFixed(4)}</span>
        </span>
      </div>
      <svg width={w} height={h} style={{ display: "block" }}>
        <line
          x1={0}
          y1={h - 1}
          x2={w}
          y2={h - 1}
          stroke="#1a1a1a"
          strokeWidth={1}
        />
        <line x1={0} y1={0} x2={0} y2={h} stroke="#1a1a1a" strokeWidth={1} />
        {points.length > 1 && (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke="#4ade80"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>
    </div>
  );
};

// --- Denoising Text ---
const DenoisingText = ({ text, progress, as: Tag = "span", style = {} }) => {
  const [displayed, setDisplayed] = useState(text);
  const chars = "█▓▒░╬╫╪┼╳※¤◊◈⬡⬢⎔⏣";

  useEffect(() => {
    if (progress >= 1) {
      setDisplayed(text);
      return;
    }
    const arr = text.split("").map((ch, i) => {
      if (ch === " ") return " ";
      const threshold = i / text.length;
      if (progress > threshold + 0.3) return ch;
      if (progress > threshold) {
        return Math.random() > 0.5
          ? ch
          : chars[Math.floor(Math.random() * chars.length)];
      }
      return chars[Math.floor(Math.random() * chars.length)];
    });
    setDisplayed(arr.join(""));
  }, [text, progress]);

  return <Tag style={style}>{displayed}</Tag>;
};

// --- Scroll-triggered section ---
const Section = ({ children, id, style = {} }) => {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { threshold: 0.1 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section
      ref={ref}
      id={id}
      style={{
        ...style,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(40px)",
        transition: "opacity 0.8s ease, transform 0.8s ease",
      }}
    >
      {children}
    </section>
  );
};

// --- Project Card ---
const ProjectCard = ({ project, index }) => {
  const [hovered, setHovered] = useState(false);
  const [step, setStep] = useState(project.defaultStep || 0);
  const stages = ["concept", "prototype", "shipped"];

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? "rgba(226,181,90,0.04)" : "rgba(255,255,255,0.02)",
        border: `1px solid ${hovered ? "#e2b55a33" : "#1a1a1a"}`,
        borderRadius: 8,
        padding: "28px 24px",
        transition: "all 0.4s ease",
        cursor: "default",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Grain overlay on card */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E")`,
          pointerEvents: "none",
          opacity: hovered ? 0 : 0.5,
          transition: "opacity 0.4s ease",
        }}
      />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 12,
        }}
      >
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#e2b55a",
            letterSpacing: 2,
          }}
        >
          SAMPLE {String(index + 1).padStart(3, "0")}
        </span>
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: "#333",
            padding: "2px 6px",
            border: "1px solid #1a1a1a",
            borderRadius: 3,
          }}
        >
          {project.year}
        </span>
      </div>

      <h3
        style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: 22,
          fontWeight: 600,
          color: "#e8e8e8",
          margin: "0 0 8px 0",
          letterSpacing: "-0.02em",
        }}
      >
        {project.name}
      </h3>

      <p
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 13,
          color: "#666",
          lineHeight: 1.6,
          margin: "0 0 16px 0",
        }}
      >
        {project.description}
      </p>

      {/* Training metrics */}
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: "#444",
          padding: "10px 12px",
          background: "rgba(0,0,0,0.3)",
          borderRadius: 4,
          marginBottom: 16,
          lineHeight: 1.8,
        }}
      >
        <span>
          loss: <span style={{ color: "#4ade80" }}>{project.metrics.loss}</span>
        </span>
        {"  |  "}
        <span>
          acc: <span style={{ color: "#4ade80" }}>{project.metrics.acc}</span>
        </span>
        {"  |  "}
        <span>
          epochs:{" "}
          <span style={{ color: "#e2b55a" }}>{project.metrics.epochs}</span>
        </span>
        <br />
        <span>
          lr: <span style={{ color: "#888" }}>{project.metrics.lr}</span>
        </span>
        {"  |  "}
        <span>
          batch_size:{" "}
          <span style={{ color: "#888" }}>{project.metrics.batch}</span>
        </span>
      </div>

      {/* Generation steps slider */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 6,
          }}
        >
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              color: "#555",
              textTransform: "uppercase",
              letterSpacing: 1,
            }}
          >
            generation step
          </span>
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              color: "#e2b55a",
            }}
          >
            {stages[step]}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={2}
          value={step}
          onChange={(e) => setStep(parseInt(e.target.value))}
          style={{
            width: "100%",
            accentColor: "#e2b55a",
            height: 2,
            cursor: "pointer",
          }}
        />
      </div>

      {/* Tags */}
      <div
        style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 14 }}
      >
        {project.tags.map((tag) => (
          <span
            key={tag}
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              color: "#555",
              border: "1px solid #222",
              padding: "3px 8px",
              borderRadius: 3,
              letterSpacing: 0.5,
            }}
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
};

// --- Interactive Loss Curve (Journey) ---
const LossCurve = () => {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const w = 700;
  const h = 280;
  const padding = { top: 30, right: 30, bottom: 40, left: 50 };

  const events = [
    { t: 0.0, label: "Started learning to code", loss: 0.95, year: "2018" },
    { t: 0.14, label: "Began CS degree at VIT Bhopal", loss: 0.80, year: "2020" },
    { t: 0.28, label: "Deep dive into Python & ML", loss: 0.62, year: "2021" },
    { t: 0.38, label: "First ML projects — exploring AI & CV", loss: 0.50, year: "2022" },
    { t: 0.48, label: "ML Intern at Learn and Empower", loss: 0.38, year: "2023" },
    { t: 0.56, label: "Loss spike — navigating job market", loss: 0.48, year: "2023" },
    { t: 0.66, label: "Trainee at KPIT — systems programming", loss: 0.30, year: "2024" },
    { t: 0.78, label: "Joined 314e as Data Science Engineer", loss: 0.15, year: "2024" },
    { t: 0.90, label: "Building AI models in production", loss: 0.08, year: "2025" },
    { t: 1.0, label: "Currently training...", loss: 0.04, year: "2025" },
  ];

  const getX = (t) => padding.left + t * (w - padding.left - padding.right);
  const getY = (loss) =>
    padding.top + (1 - loss) * (h - padding.top - padding.bottom);

  const curvePath = events
    .map((e, i) => `${i === 0 ? "M" : "L"} ${getX(e.t)} ${getY(e.loss)}`)
    .join(" ");

  return (
    <div style={{ width: "100%", overflowX: "auto" }}>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        style={{ width: "100%", maxWidth: w, display: "block", margin: "0 auto" }}
      >
        {/* Grid lines */}
        {[0.2, 0.4, 0.6, 0.8].map((v) => (
          <line
            key={v}
            x1={padding.left}
            y1={getY(v)}
            x2={w - padding.right}
            y2={getY(v)}
            stroke="#111"
            strokeWidth={0.5}
            strokeDasharray="4 4"
          />
        ))}

        {/* Axes */}
        <line
          x1={padding.left}
          y1={h - padding.bottom}
          x2={w - padding.right}
          y2={h - padding.bottom}
          stroke="#222"
          strokeWidth={1}
        />
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={h - padding.bottom}
          stroke="#222"
          strokeWidth={1}
        />

        {/* Axis labels */}
        <text
          x={padding.left - 10}
          y={padding.top + 4}
          fill="#333"
          fontSize={8}
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="end"
        >
          1.0
        </text>
        <text
          x={padding.left - 10}
          y={h - padding.bottom + 4}
          fill="#333"
          fontSize={8}
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="end"
        >
          0.0
        </text>
        <text
          x={w / 2}
          y={h - 8}
          fill="#333"
          fontSize={9}
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="middle"
        >
          time →
        </text>
        <text
          x={12}
          y={h / 2}
          fill="#333"
          fontSize={9}
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="middle"
          transform={`rotate(-90, 12, ${h / 2})`}
        >
          loss
        </text>

        {/* Curve */}
        <path
          d={curvePath}
          fill="none"
          stroke="#4ade80"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Glow */}
        <path
          d={curvePath}
          fill="none"
          stroke="#4ade80"
          strokeWidth={6}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.15}
        />

        {/* Points */}
        {events.map((e, i) => (
          <g key={i}>
            <circle
              cx={getX(e.t)}
              cy={getY(e.loss)}
              r={hoveredPoint === i ? 6 : 4}
              fill={hoveredPoint === i ? "#e2b55a" : "#4ade80"}
              stroke={hoveredPoint === i ? "#e2b55a" : "none"}
              strokeWidth={2}
              style={{ cursor: "pointer", transition: "all 0.2s ease" }}
              onMouseEnter={() => setHoveredPoint(i)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
            {hoveredPoint === i && (
              <g>
                <rect
                  x={getX(e.t) - 110}
                  y={getY(e.loss) - 46}
                  width={220}
                  height={36}
                  rx={4}
                  fill="rgba(0,0,0,0.9)"
                  stroke="#e2b55a33"
                  strokeWidth={1}
                />
                <text
                  x={getX(e.t)}
                  y={getY(e.loss) - 30}
                  fill="#e2b55a"
                  fontSize={9}
                  fontFamily="'JetBrains Mono', monospace"
                  textAnchor="middle"
                >
                  [{e.year}] {e.label}
                </text>
                <text
                  x={getX(e.t)}
                  y={getY(e.loss) - 18}
                  fill="#4ade80"
                  fontSize={8}
                  fontFamily="'JetBrains Mono', monospace"
                  textAnchor="middle"
                >
                  loss: {e.loss.toFixed(2)}
                </text>
              </g>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
};

// --- Architecture Diagram ---
const ArchitectureDiagram = () => {
  const layers = [
    { name: "Input", neurons: ["Curiosity", "Drive", "Vision"], color: "#e2b55a" },
    { name: "Conv2d", neurons: ["Python", "C++", "JavaScript"], color: "#4ade80" },
    { name: "Dense", neurons: ["ML", "Deep Learning", "NLP"], color: "#60a5fa" },
    { name: "Dense", neurons: ["Data Science", "CV", "Systems"], color: "#c084fc" },
    { name: "Output", neurons: ["AI Engineer"], color: "#f472b6" },
  ];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        flexWrap: "wrap",
        padding: "20px 0",
      }}
    >
      {layers.map((layer, li) => (
        <div key={li} style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
                color: layer.color,
                marginBottom: 8,
                letterSpacing: 1,
                textTransform: "uppercase",
              }}
            >
              {layer.name}
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                alignItems: "center",
              }}
            >
              {layer.neurons.map((n, ni) => (
                <div
                  key={ni}
                  style={{
                    width: "fit-content",
                    padding: "6px 14px",
                    border: `1px solid ${layer.color}44`,
                    borderRadius: 4,
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 11,
                    color: layer.color,
                    background: `${layer.color}08`,
                    whiteSpace: "nowrap",
                  }}
                >
                  {n}
                </div>
              ))}
            </div>
          </div>
          {li < layers.length - 1 && (
            <span style={{ color: "#222", fontSize: 20 }}>→</span>
          )}
        </div>
      ))}
    </div>
  );
};

// --- MAIN APP ---
export default function Portfolio() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [heroDenoised, setHeroDenoised] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const total =
        document.documentElement.scrollHeight - window.innerHeight;
      const progress = Math.min(window.scrollY / total, 1);
      setScrollProgress(progress);
      const heroP = Math.min(window.scrollY / (window.innerHeight * 0.25), 1);
      setHeroDenoised(heroP);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const noiseOpacity = Math.max(0.6 - scrollProgress * 1.2, 0);

  const projects = [
    {
      name: "Bloospace",
      description:
        "AI-powered mobile app that extracts and organizes actionable content from social media videos. Entity-first architecture identifies items, places, products, recipes, and people. Vibecoded.",
      year: "2026",
      tags: ["AI/ML", "Mobile", "Gemini", "Claude", "Supabase"],
      metrics: { loss: "0.034", acc: "97.2%", epochs: "847", lr: "0.001", batch: "life" },
      defaultStep: 1,
    },
    {
      name: "Autonomous Driving RL",
      description:
        "Reinforcement learning simulation for autonomous driving using Deep Q-Networks. Trained agents to navigate complex traffic scenarios with reward shaping.",
      year: "2021",
      tags: ["PyTorch", "DQN", "Simulation", "Python"],
      metrics: { loss: "0.112", acc: "89.4%", epochs: "2400", lr: "0.0003", batch: "256" },
    },
    {
      name: "Real-time Sign Language Detection",
      description:
        "Explored AI models for real-time sign language detection and translation from video streams. Investigated multiple CV approaches for hand gesture recognition.",
      year: "2024",
      tags: ["Computer Vision", "Video Processing", "TensorFlow"],
      metrics: { loss: "0.067", acc: "94.1%", epochs: "560", lr: "0.0005", batch: "64" },
    },
    {
      name: "AI Content Pipeline",
      description:
        "End-to-end content creation system leveraging AI for research, scriptwriting, and visual generation. Built for Instagram Reels and YouTube shorts about tech and AI.",
      year: "2025",
      tags: ["Content", "AI Generation", "FFmpeg", "Automation"],
      metrics: { loss: "0.021", acc: "98.6%", epochs: "365", lr: "0.002", batch: "daily" },
      defaultStep: 1,
    },
    {
      name: "The Simulation Space",
      description:
        "Browser-based geospatial intelligence platform combining a 3D interactive globe with real-time data from 50+ sources, a multi-panel intelligence dashboard, an AI-powered research terminal, and a multi-agent simulation engine for geopolitical scenario modeling. 56 toggleable data layers monitoring maritime, aviation, satellites, weather, seismic, and infrastructure. Vibecoded.",
      year: "2025",
      tags: ["Next.js", "CesiumJS", "deck.gl", "Gemini", "CAMEL-AI", "TypeScript"],
      metrics: { loss: "0.018", acc: "99.1%", epochs: "1200", lr: "0.0001", batch: "realtime" },
      defaultStep: 1,
    },
    {
      name: "SketchToReal",
      description:
        "Real-time generative canvas where users draw sketches on one side of a split-view while AI instantly transforms them into photorealistic images on the other — updating live with under 500ms latency. Powered by few-step diffusion models (SDXL Turbo / FLUX.1 Schnell) conditioned via ControlNet, optimized through StreamDiffusion for 30+ FPS on a single GPU, with optional one-click sketch-to-3D mesh generation via Stable Fast 3D. All open-source, served over WebSockets.",
      year: "2026",
      tags: ["Diffusion Models", "ControlNet", "StreamDiffusion", "WebSockets", "3D"],
      metrics: { loss: "—", acc: "—", epochs: "0", lr: "TBD", batch: "stream" },
      defaultStep: 0,
    },
  ];

  const outputs = [
    { title: "Anthropic's 2025 Financial Deep Dive", type: "Instagram Reel", icon: "◈" },
    { title: "Building Bloospace in Public", type: "YouTube Series", icon: "◈" },
    { title: "AI Model Architectures Explained", type: "Blog Post", icon: "◈" },
    { title: "Founder Journey: From Idea to MVP", type: "Thread", icon: "◈" },
  ];

  // Inline styles for Google Fonts
  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  return (
    <div
      style={{
        background: "#050505",
        color: "#e8e8e8",
        minHeight: "100vh",
        fontFamily: "'Space Grotesk', sans-serif",
        position: "relative",
      }}
    >
      <NoiseCanvas opacity={noiseOpacity} />
      <MiniLossCurve progress={scrollProgress} />

      {/* ===== NAV ===== */}
      <nav
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10001,
          padding: "16px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: scrollProgress > 0.05 ? "rgba(5,5,5,0.8)" : "transparent",
          backdropFilter: scrollProgress > 0.05 ? "blur(12px)" : "none",
          borderBottom: scrollProgress > 0.05 ? "1px solid #111" : "1px solid transparent",
          transition: "all 0.3s ease",
        }}
      >
        <span
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            color: "#e2b55a",
            fontWeight: 600,
            letterSpacing: 1,
          }}
        >
          ronak.ai
        </span>
        <div style={{ display: "flex", gap: 28 }}>
          {["about", "experience", "projects", "architecture", "journey", "contact"].map(
            (s) => (
              <a
                key={s}
                href={`#${s}`}
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: "#555",
                  textDecoration: "none",
                  letterSpacing: 1,
                  textTransform: "uppercase",
                  transition: "color 0.2s",
                }}
                onMouseEnter={(e) => (e.target.style.color = "#e2b55a")}
                onMouseLeave={(e) => (e.target.style.color = "#555")}
              >
                {s}
              </a>
            )
          )}
          <Link
            to="/learn"
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              color: "#555",
              textDecoration: "none",
              letterSpacing: 1,
              textTransform: "uppercase",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.target.style.color = "#e2b55a")}
            onMouseLeave={(e) => (e.target.style.color = "#555")}
          >
            learn
          </Link>
        </div>
      </nav>

      {/* ===== HERO ===== */}
      <section
        style={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          padding: "0 32px",
          position: "relative",
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 32,
          }}
        >
          EPOCH 0 — INITIALIZING WEIGHTS
        </div>

        <DenoisingText
          text="RONAK"
          progress={heroDenoised}
          as="h1"
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: "clamp(48px, 10vw, 120px)",
            fontWeight: 700,
            letterSpacing: "-0.04em",
            margin: 0,
            lineHeight: 1,
            color: "#e8e8e8",
          }}
        />

        <DenoisingText
          text="AI Builder. Currently training."
          progress={Math.max(heroDenoised - 0.2, 0) * 1.25}
          as="p"
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 15,
            color: "#666",
            marginTop: 20,
            letterSpacing: 1,
          }}
        />

        <div
          style={{
            marginTop: 48,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#222",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span>scroll to train</span>
          <span
            style={{
              animation: "bounce 2s infinite",
              fontSize: 16,
            }}
          >
            ↓
          </span>
        </div>

        <style>{`
          @keyframes bounce {
            0%, 100% { transform: translateY(0); opacity: 0.3; }
            50% { transform: translateY(8px); opacity: 0.6; }
          }
        `}</style>
      </section>

      {/* ===== ABOUT ===== */}
      <Section
        id="about"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 12 — EARLY CONVERGENCE
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 24px 0",
            color: "#e8e8e8",
          }}
        >
          About
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 14,
            color: "#888",
            lineHeight: 1.9,
            margin: 0,
          }}
        >
          I'm a data science engineer and AI builder based in India, currently
          working at <span style={{ color: "#e2b55a" }}>314e Corporation</span> —
          developing and testing AI models across multiple enterprise products.
          My background spans machine learning, deep learning, NLP, and systems
          programming. I think in architectures, ship in iterations, and believe
          the best way to learn is to build relentlessly.
        </p>

        <div
          style={{
            marginTop: 40,
            padding: "16px 20px",
            border: "1px solid #1a1a1a",
            borderRadius: 6,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11,
            color: "#444",
            lineHeight: 2,
          }}
        >
          <span style={{ color: "#333" }}>$</span> cat config.yaml
          <br />
          <span style={{ color: "#e2b55a" }}>name:</span> Ronak Sharma
          <br />
          <span style={{ color: "#e2b55a" }}>role:</span> Data Science Engineer & AI Builder
          <br />
          <span style={{ color: "#e2b55a" }}>focus:</span> [AI/ML, Deep Learning, NLP]
          <br />
          <span style={{ color: "#e2b55a" }}>stack:</span> [Python, C++, TensorFlow, PyTorch, SQL]
          <br />
          <span style={{ color: "#e2b55a" }}>education:</span> VIT Bhopal (CSE, 2024)
          <br />
          <span style={{ color: "#e2b55a" }}>status:</span>{" "}
          <span style={{ color: "#4ade80" }}>currently_training</span>
          <br />
          <span style={{ color: "#e2b55a" }}>learning_rate:</span> 0.001 (but adaptive)
        </div>
      </Section>

      {/* ===== EXPERIENCE / PRE-TRAINING ===== */}
      <Section
        id="experience"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 24 — PRE-TRAINING
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 12px 0",
            color: "#e8e8e8",
          }}
        >
          Pre-training
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#444",
            marginBottom: 40,
          }}
        >
          Large-scale foundational training. The base model before fine-tuning.
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {[
            {
              company: "314e Corporation",
              role: "Associate Data Science Engineer",
              period: "July 2024 — Present",
              location: "Bangalore, India",
              status: "active",
              objectives: [
                "Testing and evaluating AI models across multiple healthcare and enterprise products",
                "Developing and fine-tuning machine learning pipelines for production deployment",
                "Collaborating with cross-functional teams to integrate AI capabilities into existing products",
                "Building model evaluation frameworks and performance benchmarking tools",
              ],
              capabilities: ["Data Science", "AI/ML", "Model Evaluation", "Production ML", "NLP"],
              metrics: { dataset: "enterprise-scale", compute: "cloud-GPU", duration: "1+ year" },
            },
            {
              company: "KPIT Technologies",
              role: "Trainee",
              period: "January 2024 — July 2024",
              location: "Bangalore, India",
              status: "completed",
              objectives: [
                "Received intensive training on C and Modern C++ programming paradigms",
                "Learned and implemented various software design patterns",
                "Applied systems programming concepts in automotive technology context",
              ],
              capabilities: ["C", "Modern C++", "Design Patterns", "Systems Programming"],
              metrics: { dataset: "automotive", compute: "embedded", duration: "6 months" },
            },
            {
              company: "Learn and Empower Pvt Ltd",
              role: "Machine Learning Intern",
              period: "November 2023 — April 2024",
              location: "Remote",
              status: "completed",
              objectives: [
                "Designed and developed ML, Deep Learning, and NLP models from scratch",
                "Integrated trained models into production-ready systems",
                "Worked on data preprocessing pipelines and feature engineering",
              ],
              capabilities: ["Machine Learning", "Deep Learning", "NLP", "Model Integration"],
              metrics: { dataset: "varied", compute: "single-GPU", duration: "6 months" },
            },
          ].map((job, i) => (
            <div
              key={i}
              style={{
                border: "1px solid #1a1a1a",
                borderRadius: 8,
                padding: "28px 24px",
                background: "rgba(255,255,255,0.02)",
                position: "relative",
                overflow: "hidden",
                transition: "border-color 0.3s ease",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#e2b55a22")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "#1a1a1a")}
            >
              {/* Active indicator */}
              {job.status === "active" && (
                <div
                  style={{
                    position: "absolute",
                    top: 28,
                    right: 24,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "#4ade80",
                      boxShadow: "0 0 8px #4ade8066",
                      animation: "pulse-dot 2s infinite",
                    }}
                  />
                  <span
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 9,
                      color: "#4ade80",
                      letterSpacing: 1,
                      textTransform: "uppercase",
                    }}
                  >
                    Training
                  </span>
                </div>
              )}

              <style>{`
                @keyframes pulse-dot {
                  0%, 100% { opacity: 1; }
                  50% { opacity: 0.4; }
                }
              `}</style>

              {/* Period */}
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: "#e2b55a",
                  letterSpacing: 2,
                  marginBottom: 12,
                }}
              >
                {job.period}
              </div>

              {/* Company & Role */}
              <h3
                style={{
                  fontFamily: "'Space Grotesk', sans-serif",
                  fontSize: 22,
                  fontWeight: 600,
                  color: "#e8e8e8",
                  margin: "0 0 4px 0",
                  letterSpacing: "-0.02em",
                }}
              >
                {job.company}
              </h3>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 12,
                  color: "#666",
                  marginBottom: 4,
                }}
              >
                {job.role}
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: "#444",
                  marginBottom: 20,
                }}
              >
                ↳ {job.location}
              </div>

              {/* Training objectives as log */}
              <div
                style={{
                  padding: "14px 16px",
                  background: "rgba(0,0,0,0.3)",
                  borderRadius: 4,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: "#555",
                  lineHeight: 2,
                  marginBottom: 16,
                }}
              >
                <div style={{ color: "#333", marginBottom: 4, fontSize: 9, letterSpacing: 1 }}>
                  TRAINING OBJECTIVES
                </div>
                {job.objectives.map((obj, j) => (
                  <div key={j}>
                    <span style={{ color: "#4ade80" }}>✓</span>{" "}
                    <span style={{ color: "#777" }}>{obj}</span>
                  </div>
                ))}
              </div>

              {/* Training config */}
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  color: "#444",
                  marginBottom: 14,
                }}
              >
                <span>
                  dataset: <span style={{ color: "#e2b55a" }}>{job.metrics.dataset}</span>
                </span>
                {"  |  "}
                <span>
                  compute: <span style={{ color: "#e2b55a" }}>{job.metrics.compute}</span>
                </span>
                {"  |  "}
                <span>
                  duration: <span style={{ color: "#e2b55a" }}>{job.metrics.duration}</span>
                </span>
              </div>

              {/* Capabilities acquired */}
              <div
                style={{ display: "flex", flexWrap: "wrap", gap: 6 }}
              >
                {job.capabilities.map((cap) => (
                  <span
                    key={cap}
                    style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 9,
                      color: "#4ade80",
                      border: "1px solid #4ade8022",
                      background: "#4ade8008",
                      padding: "3px 8px",
                      borderRadius: 3,
                      letterSpacing: 0.5,
                    }}
                  >
                    {cap}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Pre-training summary */}
        <div
          style={{
            marginTop: 24,
            padding: "14px 20px",
            border: "1px solid #1a1a1a",
            borderRadius: 6,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#444",
            lineHeight: 2,
          }}
        >
          <span style={{ color: "#333" }}>$</span> pretrain.summary()
          <br />
          base_capabilities: <span style={{ color: "#4ade80" }}>loaded</span>
          <br />
          ready_for_finetuning: <span style={{ color: "#4ade80" }}>True</span>
          <br />
          transfer_learning: <span style={{ color: "#e2b55a" }}>enterprise → builder</span>
        </div>
      </Section>

      {/* ===== PROJECTS ===== */}
      <Section
        id="projects"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 34 — FINE-TUNING DATA
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 12px 0",
            color: "#e8e8e8",
          }}
        >
          Projects
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#444",
            marginBottom: 40,
          }}
        >
          The samples I chose to specialize on.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: 20,
          }}
        >
          {projects.map((p, i) => (
            <ProjectCard key={i} project={p} index={i} />
          ))}
        </div>
      </Section>

      {/* ===== ARCHITECTURE ===== */}
      <Section
        id="architecture"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 56 — MODEL ARCHITECTURE
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 12px 0",
            color: "#e8e8e8",
          }}
        >
          Architecture
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#444",
            marginBottom: 40,
          }}
        >
          The layers that make up this model.
        </p>

        <div
          style={{
            padding: "32px 20px",
            border: "1px solid #1a1a1a",
            borderRadius: 8,
            background: "rgba(255,255,255,0.01)",
            overflowX: "auto",
          }}
        >
          <ArchitectureDiagram />
        </div>

        {/* Model summary */}
        <div
          style={{
            marginTop: 24,
            padding: "16px 20px",
            border: "1px solid #1a1a1a",
            borderRadius: 6,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#444",
            lineHeight: 2,
          }}
        >
          <span style={{ color: "#333" }}>$</span> model.summary()
          <br />
          Total params: <span style={{ color: "#4ade80" }}>∞ (growing)</span>
          <br />
          Trainable: <span style={{ color: "#4ade80" }}>Yes</span>
          <br />
          Optimizer: <span style={{ color: "#e2b55a" }}>AdamW(curiosity, caffeine)</span>
          <br />
          Regularization: <span style={{ color: "#e2b55a" }}>Dropout(burnout, 0.1)</span>
          <br />
          Activation: <span style={{ color: "#e2b55a" }}>ReLU(passion)</span>
        </div>
      </Section>

      {/* ===== JOURNEY / LOSS CURVE ===== */}
      <Section
        id="journey"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 78 — CONVERGENCE ANALYSIS
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 12px 0",
            color: "#e8e8e8",
          }}
        >
          Loss Curve
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#444",
            marginBottom: 40,
          }}
        >
          Hover over the data points to trace the journey.
        </p>

        <div
          style={{
            padding: "24px",
            border: "1px solid #1a1a1a",
            borderRadius: 8,
            background: "rgba(255,255,255,0.01)",
          }}
        >
          <LossCurve />
        </div>
      </Section>

      {/* ===== OUTPUTS ===== */}
      <Section
        id="outputs"
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "120px 32px",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 92 — MODEL OUTPUTS
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 40,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 12px 0",
            color: "#e8e8e8",
          }}
        >
          Inference Results
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            color: "#444",
            marginBottom: 40,
          }}
        >
          Now that the model is trained, here's what it generates.
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {outputs.map((o, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "18px 22px",
                border: "1px solid #1a1a1a",
                borderRadius: 6,
                background: "rgba(255,255,255,0.01)",
                cursor: "pointer",
                transition: "all 0.3s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#e2b55a33";
                e.currentTarget.style.background = "rgba(226,181,90,0.03)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#1a1a1a";
                e.currentTarget.style.background = "rgba(255,255,255,0.01)";
              }}
            >
              <div>
                <div
                  style={{
                    fontFamily: "'Space Grotesk', sans-serif",
                    fontSize: 16,
                    fontWeight: 500,
                    color: "#ccc",
                    marginBottom: 4,
                  }}
                >
                  {o.icon} {o.title}
                </div>
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 10,
                    color: "#444",
                  }}
                >
                  {o.type}
                </div>
              </div>
              <span style={{ color: "#333", fontSize: 18 }}>→</span>
            </div>
          ))}
        </div>
      </Section>

      {/* ===== CONTACT ===== */}
      <Section
        id="contact"
        style={{
          maxWidth: 700,
          margin: "0 auto",
          padding: "120px 32px 80px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#333",
            letterSpacing: 3,
            marginBottom: 40,
          }}
        >
          EPOCH 100 — INFERENCE READY
        </div>

        <h2
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 48,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            margin: "0 0 16px 0",
            color: "#e8e8e8",
          }}
        >
          Run Inference
        </h2>

        <p
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 13,
            color: "#555",
            marginBottom: 48,
          }}
        >
          The model is converged. Submit your prompt.
        </p>

        {/* API-style contact */}
        <div
          style={{
            textAlign: "left",
            padding: "24px 28px",
            border: "1px solid #1a1a1a",
            borderRadius: 8,
            background: "rgba(255,255,255,0.02)",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
            lineHeight: 2.2,
            color: "#555",
            maxWidth: 480,
            margin: "0 auto",
          }}
        >
          <span style={{ color: "#4ade80" }}>POST</span>{" "}
          <span style={{ color: "#888" }}>/api/collaborate</span>
          <br />
          <span style={{ color: "#333" }}>Content-Type:</span> application/json
          <br />
          <br />
          <span style={{ color: "#888" }}>{"{"}</span>
          <br />
          {"  "}
          <span style={{ color: "#e2b55a" }}>"from"</span>
          <span style={{ color: "#888" }}>: </span>
          <span style={{ color: "#4ade80" }}>"you"</span>,
          <br />
          {"  "}
          <span style={{ color: "#e2b55a" }}>"subject"</span>
          <span style={{ color: "#888" }}>: </span>
          <span style={{ color: "#4ade80" }}>"let's build something"</span>,
          <br />
          {"  "}
          <span style={{ color: "#e2b55a" }}>"urgency"</span>
          <span style={{ color: "#888" }}>: </span>
          <span style={{ color: "#4ade80" }}>"high"</span>
          <br />
          <span style={{ color: "#888" }}>{"}"}</span>
        </div>

        {/* Contact Links */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            marginTop: 40,
          }}
        >
          {[
            { label: "Email", href: "#" },
            { label: "Twitter/X", href: "#" },
            { label: "GitHub", href: "https://github.com/Ronaknowal" },
            { label: "Instagram", href: "#" },
            { label: "LinkedIn", href: "https://www.linkedin.com/in/ronak-sharma-a6455a1b5/" },
          ].map((link) => (
            <a
              key={link.label}
              href={link.href}
              target={link.href !== "#" ? "_blank" : undefined}
              rel={link.href !== "#" ? "noopener noreferrer" : undefined}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                color: "#444",
                textDecoration: "none",
                transition: "color 0.2s",
                letterSpacing: 0.5,
              }}
              onMouseEnter={(e) => (e.target.style.color = "#e2b55a")}
              onMouseLeave={(e) => (e.target.style.color = "#444")}
            >
              {link.label}
            </a>
          ))}
        </div>
      </Section>

      {/* ===== FOOTER ===== */}
      <footer
        style={{
          padding: "40px 32px",
          textAlign: "center",
          borderTop: "1px solid #0a0a0a",
        }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#222",
            lineHeight: 2,
          }}
        >
          model_version: 2025.02 | framework: life
          <br />
          built with curiosity, caffeine, and gradient descent
          <br />
          <span style={{ color: "#333" }}>
            © {new Date().getFullYear()} ronak.ai — still training
          </span>
        </div>
      </footer>
    </div>
  );
}
