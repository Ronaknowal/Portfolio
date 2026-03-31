import { useState, useCallback } from "react";
import { colors, fonts } from "../styles";

// Simple 2-layer network: x → (W1,b1) → ReLU → (W2,b2) → output → loss
// Fixed weights for deterministic visualization
const NET = {
  x: 1.5,
  target: 0.8,
  W1: 0.6,
  b1: 0.1,
  W2: -0.4,
  b2: 0.3,
};

function forwardPass({ x, W1, b1, W2, b2, target }) {
  const z1 = W1 * x + b1;        // pre-activation
  const a1 = Math.max(0, z1);    // ReLU
  const z2 = W2 * a1 + b2;       // output
  const loss = 0.5 * (z2 - target) ** 2; // MSE / 2
  return { z1, a1, z2, loss };
}

function backwardPass({ x, W1, W2, target }, { z1, a1, z2 }) {
  const dL_dz2 = z2 - target;
  const dL_dW2 = dL_dz2 * a1;
  const dL_db2 = dL_dz2;
  const dL_da1 = dL_dz2 * W2;
  const dL_dz1 = dL_da1 * (z1 > 0 ? 1 : 0); // ReLU grad
  const dL_dW1 = dL_dz1 * x;
  const dL_db1 = dL_dz1;
  return { dL_dz2, dL_dW2, dL_db2, dL_da1, dL_dz1, dL_dW1, dL_db1 };
}

const STAGES = ["idle", "forward", "backward"];

export default function BackpropVisualizer() {
  const [stage, setStage] = useState("idle");
  const [step, setStep] = useState(0); // animation step within stage

  const fwd = forwardPass(NET);
  const bwd = backwardPass(NET, fwd);

  const maxForwardSteps = 4; // x → z1 → a1 → z2 → loss
  const maxBackwardSteps = 4; // dL/dz2 → dL/da1 → dL/dz1 → dL/dW1

  const handleForward = useCallback(() => {
    setStage("forward");
    setStep(0);
  }, []);

  const handleBackward = useCallback(() => {
    setStage("backward");
    setStep(0);
  }, []);

  const handleStep = useCallback(() => {
    const max = stage === "forward" ? maxForwardSteps : maxBackwardSteps;
    if (step < max) {
      setStep((s) => s + 1);
    }
  }, [stage, step]);

  const handleReset = useCallback(() => {
    setStage("idle");
    setStep(0);
  }, []);

  // Node rendering helper
  const Node = ({ label, value, gradient, x, y, active, gradActive, isLoss }) => {
    const showValue = stage === "forward" ? active : true;
    const showGrad = stage === "backward" && gradActive;

    return (
      <g>
        {/* Node circle */}
        <circle
          cx={x}
          cy={y}
          r={28}
          fill={
            showGrad ? `${colors.green}15` :
            (active && stage === "forward") ? `${colors.gold}15` :
            "rgba(255,255,255,0.02)"
          }
          stroke={
            showGrad ? colors.green :
            (active && stage === "forward") ? colors.gold :
            colors.border
          }
          strokeWidth={active || showGrad ? 1.5 : 1}
          style={{ transition: "all 0.3s ease" }}
        />

        {/* Label */}
        <text
          x={x}
          y={y - 8}
          textAnchor="middle"
          fill={colors.textMuted}
          fontFamily={fonts.mono}
          fontSize={9}
        >
          {label}
        </text>

        {/* Value */}
        {showValue && value !== undefined && (
          <text
            x={x}
            y={y + 8}
            textAnchor="middle"
            fill={isLoss ? "#f87171" : colors.gold}
            fontFamily={fonts.mono}
            fontSize={10}
            fontWeight={600}
          >
            {typeof value === "number" ? value.toFixed(3) : value}
          </text>
        )}

        {/* Gradient below node */}
        {showGrad && gradient !== undefined && (
          <text
            x={x}
            y={y + 46}
            textAnchor="middle"
            fill={colors.green}
            fontFamily={fonts.mono}
            fontSize={9}
          >
            ∂={typeof gradient === "number" ? gradient.toFixed(3) : gradient}
          </text>
        )}
      </g>
    );
  };

  // Edge rendering helper
  const Edge = ({ x1, y1, x2, y2, label, active, gradActive, gradLabel }) => {
    const isForwardActive = stage === "forward" && active;
    const isBackActive = stage === "backward" && gradActive;

    return (
      <g>
        <line
          x1={x1 + 28}
          y1={y1}
          x2={x2 - 28}
          y2={y2}
          stroke={
            isBackActive ? colors.green :
            isForwardActive ? colors.gold :
            colors.border
          }
          strokeWidth={isForwardActive || isBackActive ? 1.5 : 0.5}
          strokeDasharray={isBackActive ? "4,3" : "none"}
          style={{ transition: "all 0.3s ease" }}
        />

        {/* Weight label on edge */}
        {label && (
          <text
            x={(x1 + x2) / 2}
            y={(y1 + y2) / 2 - 10}
            textAnchor="middle"
            fill={colors.textDark}
            fontFamily={fonts.mono}
            fontSize={8}
          >
            {label}
          </text>
        )}

        {/* Gradient label on edge (backward) */}
        {isBackActive && gradLabel && (
          <text
            x={(x1 + x2) / 2}
            y={(y1 + y2) / 2 + 14}
            textAnchor="middle"
            fill={colors.green}
            fontFamily={fonts.mono}
            fontSize={8}
          >
            {gradLabel}
          </text>
        )}

        {/* Direction arrow */}
        {(isForwardActive || isBackActive) && (
          <circle
            cx={isBackActive ? x1 + 32 : x2 - 32}
            cy={y2}
            r={2}
            fill={isBackActive ? colors.green : colors.gold}
          />
        )}
      </g>
    );
  };

  // Determine visibility per step
  const fwdActive = (minStep) => stage === "forward" && step >= minStep;
  const bwdActive = (minStep) => stage === "backward" && step >= minStep;

  // Node positions
  const ny = 100; // vertical center
  const nodes = {
    x:    { x: 60,  y: ny },
    z1:   { x: 180, y: ny },
    a1:   { x: 300, y: ny },
    z2:   { x: 420, y: ny },
    loss: { x: 540, y: ny },
  };

  return (
    <div style={{
      border: `1px solid ${colors.border}`,
      borderRadius: 6,
      padding: 20,
      background: "rgba(0,0,0,0.3)",
      marginBottom: 20,
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <div style={{ fontFamily: fonts.mono, fontSize: 10, color: colors.gold, letterSpacing: 1, marginBottom: 4 }}>
            INTERACTIVE
          </div>
          <div style={{ fontFamily: fonts.sans, fontSize: 14, fontWeight: 500, color: colors.textPrimary }}>
            Backpropagation Computation Graph
          </div>
        </div>
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: colors.textDim }}>
          {stage === "idle" && "click forward to start"}
          {stage === "forward" && `forward pass — step ${step}/${maxForwardSteps}`}
          {stage === "backward" && `backward pass — step ${step}/${maxBackwardSteps}`}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button
          onClick={handleForward}
          disabled={stage === "forward"}
          style={{
            padding: "5px 14px",
            borderRadius: 3,
            border: `1px solid ${stage !== "forward" ? `${colors.gold}44` : colors.border}`,
            background: stage === "forward" ? `${colors.gold}11` : "transparent",
            color: stage !== "forward" ? colors.gold : colors.textDim,
            fontFamily: fonts.mono,
            fontSize: 10,
            cursor: stage !== "forward" ? "pointer" : "default",
          }}
        >
          Forward
        </button>
        <button
          onClick={handleBackward}
          disabled={stage !== "forward" || step < maxForwardSteps}
          style={{
            padding: "5px 14px",
            borderRadius: 3,
            border: `1px solid ${(stage === "forward" && step >= maxForwardSteps) ? `${colors.green}44` : colors.border}`,
            background: stage === "backward" ? `${colors.green}11` : "transparent",
            color: (stage === "forward" && step >= maxForwardSteps) ? colors.green : colors.textDim,
            fontFamily: fonts.mono,
            fontSize: 10,
            cursor: (stage === "forward" && step >= maxForwardSteps) ? "pointer" : "default",
          }}
        >
          Backward
        </button>
        <button
          onClick={handleStep}
          disabled={stage === "idle" || (stage === "forward" && step >= maxForwardSteps) || (stage === "backward" && step >= maxBackwardSteps)}
          style={{
            padding: "5px 14px",
            borderRadius: 3,
            border: `1px solid ${colors.border}`,
            background: "transparent",
            color: stage !== "idle" ? colors.textSecondary : colors.textDim,
            fontFamily: fonts.mono,
            fontSize: 10,
            cursor: stage !== "idle" ? "pointer" : "default",
          }}
        >
          Step →
        </button>
        <button
          onClick={handleReset}
          style={{
            padding: "5px 14px",
            borderRadius: 3,
            border: `1px solid ${colors.border}`,
            background: "transparent",
            color: colors.textMuted,
            fontFamily: fonts.mono,
            fontSize: 10,
            cursor: "pointer",
          }}
        >
          Reset
        </button>
      </div>

      {/* Graph SVG */}
      <div style={{ overflowX: "auto" }}>
        <svg width={600} height={200} style={{ display: "block" }}>
          {/* Edges */}
          <Edge
            x1={nodes.x.x} y1={nodes.x.y} x2={nodes.z1.x} y2={nodes.z1.y}
            label={`W1=${NET.W1}`}
            active={fwdActive(1)}
            gradActive={bwdActive(3)}
            gradLabel={`∂W1=${bwd.dL_dW1.toFixed(3)}`}
          />
          <Edge
            x1={nodes.z1.x} y1={nodes.z1.y} x2={nodes.a1.x} y2={nodes.a1.y}
            label="ReLU"
            active={fwdActive(2)}
            gradActive={bwdActive(2)}
            gradLabel={fwd.z1 > 0 ? "×1" : "×0"}
          />
          <Edge
            x1={nodes.a1.x} y1={nodes.a1.y} x2={nodes.z2.x} y2={nodes.z2.y}
            label={`W2=${NET.W2}`}
            active={fwdActive(3)}
            gradActive={bwdActive(1)}
            gradLabel={`∂W2=${bwd.dL_dW2.toFixed(3)}`}
          />
          <Edge
            x1={nodes.z2.x} y1={nodes.z2.y} x2={nodes.loss.x} y2={nodes.loss.y}
            label="MSE"
            active={fwdActive(4)}
            gradActive={bwdActive(0)}
            gradLabel={`∂=${bwd.dL_dz2.toFixed(3)}`}
          />

          {/* Nodes */}
          <Node
            label="x" value={NET.x}
            x={nodes.x.x} y={nodes.x.y}
            active={fwdActive(0)}
            gradActive={false}
          />
          <Node
            label="z1" value={fwdActive(1) ? fwd.z1 : undefined}
            gradient={bwd.dL_dz1}
            x={nodes.z1.x} y={nodes.z1.y}
            active={fwdActive(1)}
            gradActive={bwdActive(3)}
          />
          <Node
            label="a1" value={fwdActive(2) ? fwd.a1 : undefined}
            gradient={bwd.dL_da1}
            x={nodes.a1.x} y={nodes.a1.y}
            active={fwdActive(2)}
            gradActive={bwdActive(2)}
          />
          <Node
            label="z2" value={fwdActive(3) ? fwd.z2 : undefined}
            gradient={bwd.dL_dz2}
            x={nodes.z2.x} y={nodes.z2.y}
            active={fwdActive(3)}
            gradActive={bwdActive(1)}
          />
          <Node
            label="loss" value={fwdActive(4) ? fwd.loss : undefined}
            x={nodes.loss.x} y={nodes.loss.y}
            active={fwdActive(4)}
            gradActive={bwdActive(0)}
            isLoss
          />

          {/* Direction labels */}
          {stage === "forward" && (
            <text x={300} y={185} textAnchor="middle" fill={colors.gold} fontFamily={fonts.mono} fontSize={9} opacity={0.5}>
              → forward pass →
            </text>
          )}
          {stage === "backward" && (
            <text x={300} y={185} textAnchor="middle" fill={colors.green} fontFamily={fonts.mono} fontSize={9} opacity={0.5}>
              ← backward pass ←
            </text>
          )}
        </svg>
      </div>

      {/* Step explanation */}
      <div style={{
        marginTop: 14,
        padding: "8px 12px",
        background: "rgba(0,0,0,0.3)",
        borderRadius: 4,
        fontFamily: fonts.mono,
        fontSize: 10,
        color: colors.textDim,
        minHeight: 20,
        lineHeight: 1.6,
      }}>
        {stage === "idle" && (
          <span>Click <span style={{ color: colors.gold }}>Forward</span> to compute activations through the network, then <span style={{ color: colors.green }}>Backward</span> to propagate gradients.</span>
        )}
        {stage === "forward" && step === 0 && (
          <span>Input <span style={{ color: colors.gold }}>x = {NET.x}</span>. Click <span style={{ color: colors.textSecondary }}>Step</span> to propagate forward.</span>
        )}
        {stage === "forward" && step === 1 && (
          <span><span style={{ color: colors.gold }}>z1</span> = W1 * x + b1 = {NET.W1} * {NET.x} + {NET.b1} = <span style={{ color: colors.gold }}>{fwd.z1.toFixed(3)}</span></span>
        )}
        {stage === "forward" && step === 2 && (
          <span><span style={{ color: colors.gold }}>a1</span> = ReLU(z1) = ReLU({fwd.z1.toFixed(3)}) = <span style={{ color: colors.gold }}>{fwd.a1.toFixed(3)}</span> {fwd.z1 <= 0 && "(killed by ReLU!)"}</span>
        )}
        {stage === "forward" && step === 3 && (
          <span><span style={{ color: colors.gold }}>z2</span> = W2 * a1 + b2 = {NET.W2} * {fwd.a1.toFixed(3)} + {NET.b2} = <span style={{ color: colors.gold }}>{fwd.z2.toFixed(3)}</span></span>
        )}
        {stage === "forward" && step === 4 && (
          <span><span style={{ color: "#f87171" }}>loss</span> = 0.5 * (z2 - target)² = 0.5 * ({fwd.z2.toFixed(3)} - {NET.target})² = <span style={{ color: "#f87171" }}>{fwd.loss.toFixed(4)}</span>. Now click <span style={{ color: colors.green }}>Backward</span>.</span>
        )}
        {stage === "backward" && step === 0 && (
          <span><span style={{ color: colors.green }}>∂L/∂z2</span> = z2 - target = {fwd.z2.toFixed(3)} - {NET.target} = <span style={{ color: colors.green }}>{bwd.dL_dz2.toFixed(3)}</span>. Click Step to continue.</span>
        )}
        {stage === "backward" && step === 1 && (
          <span><span style={{ color: colors.green }}>∂L/∂W2</span> = ∂L/∂z2 * a1 = {bwd.dL_dz2.toFixed(3)} * {fwd.a1.toFixed(3)} = <span style={{ color: colors.green }}>{bwd.dL_dW2.toFixed(3)}</span> | <span style={{ color: colors.green }}>∂L/∂a1</span> = ∂L/∂z2 * W2 = <span style={{ color: colors.green }}>{bwd.dL_da1.toFixed(3)}</span></span>
        )}
        {stage === "backward" && step === 2 && (
          <span><span style={{ color: colors.green }}>∂L/∂z1</span> = ∂L/∂a1 * ReLU'(z1) = {bwd.dL_da1.toFixed(3)} * {fwd.z1 > 0 ? "1" : "0"} = <span style={{ color: colors.green }}>{bwd.dL_dz1.toFixed(3)}</span> {fwd.z1 <= 0 && "(gradient killed by ReLU!)"}</span>
        )}
        {stage === "backward" && step === 3 && (
          <span><span style={{ color: colors.green }}>∂L/∂W1</span> = ∂L/∂z1 * x = {bwd.dL_dz1.toFixed(3)} * {NET.x} = <span style={{ color: colors.green }}>{bwd.dL_dW1.toFixed(3)}</span>. All gradients computed! W1 would update: {NET.W1} - lr * {bwd.dL_dW1.toFixed(3)}</span>
        )}
        {stage === "backward" && step >= maxBackwardSteps && (
          <span>All gradients computed. Click <span style={{ color: colors.textSecondary }}>Reset</span> to start over.</span>
        )}
      </div>
    </div>
  );
}
