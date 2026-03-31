import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Portfolio from "./Portfolio";
import "./index.css";

// Lazy-load learn pages to keep initial portfolio bundle small
const LearnHub = React.lazy(() => import("./learn/LearnHub"));
const Reader = React.lazy(() => import("./learn/Reader"));

const LearnFallback = () => (
  <div style={{ background: "#050505", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#333" }}>loading weights...</span>
  </div>
);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <React.Suspense fallback={<LearnFallback />}>
        <Routes>
          <Route path="/" element={<Portfolio />} />
          <Route path="/learn" element={<LearnHub />} />
          <Route path="/learn/track/:trackId/:topicId?" element={<Reader />} />
          <Route path="/learn/topic/:topicId" element={<Reader />} />
        </Routes>
      </React.Suspense>
    </BrowserRouter>
  </React.StrictMode>
);
