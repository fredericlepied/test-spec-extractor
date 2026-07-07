import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import type { ClusterTest, ClusterEdge } from "../../types";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

interface Props {
  nodes: ClusterTest[];
  edges: ClusterEdge[];
}

export function ForceGraph({ nodes, edges }: Props) {
  const navigate = useNavigate();
  const [hover, setHover] = useState<{ node: ClusterTest; x: number; y: number } | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform((t) => ({ ...t, scale: Math.max(0.2, Math.min(5, t.scale * delta)) }));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.target === svgRef.current || (e.target as Element).tagName === "svg") {
      dragRef.current = { startX: e.clientX, startY: e.clientY, origX: transform.x, origY: transform.y };
    }
  }, [transform.x, transform.y]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    setTransform((t) => ({
      ...t,
      x: dragRef.current!.origX + (e.clientX - dragRef.current!.startX),
      y: dragRef.current!.origY + (e.clientY - dragRef.current!.startY),
    }));
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);

  return (
    <div className="relative overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 w-full h-full">
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="cursor-grab active:cursor-grabbing"
      >
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
          {edges.map((e, i) => (
            <line
              key={i}
              x1={nodes[e.sourceIdx].x}
              y1={nodes[e.sourceIdx].y}
              x2={nodes[e.targetIdx].x}
              y2={nodes[e.targetIdx].y}
              stroke={COLORS[nodes[e.sourceIdx].colorIndex]}
              strokeOpacity={0.15 + (e.score - 0.9) * 8}
              strokeWidth={1}
            />
          ))}
          {nodes.map((n, i) => (
            <circle
              key={i}
              cx={n.x}
              cy={n.y}
              r={Math.max(3, Math.min(8, 2 + n.degree))}
              fill={COLORS[n.colorIndex]}
              className="cursor-pointer"
              onMouseEnter={(e) => setHover({ node: n, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setHover(null)}
              onClick={() => navigate(`/catalog?repo=${encodeURIComponent(n.repo)}&search=${encodeURIComponent(n.desc)}`)}
            />
          ))}
        </g>
      </svg>
      {hover && (
        <div
          className="fixed z-50 px-3 py-2 rounded-lg text-xs shadow-lg pointer-events-none max-w-xs"
          style={{
            left: hover.x + 12,
            top: hover.y - 10,
            backgroundColor: "var(--color-tooltip-bg)",
            color: "var(--color-tooltip-text)",
            border: "1px solid var(--color-tooltip-border)",
          }}
        >
          <div className="font-medium truncate">{hover.node.desc}</div>
          <div className="text-gray-400">{hover.node.repo} / {hover.node.file}</div>
          {hover.node.testId && <div className="text-gray-400">OCP-{hover.node.testId}</div>}
        </div>
      )}
    </div>
  );
}
