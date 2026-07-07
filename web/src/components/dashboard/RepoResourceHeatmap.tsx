import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { HeatmapData } from "../../types";

const LIGHT_SCALE = ["#f0f9ff", "#bae6fd", "#38bdf8", "#0284c7", "#1e3a8a"];
const DARK_SCALE = ["#0c1929", "#172554", "#1e40af", "#3b82f6", "#93c5fd"];

function getColor(count: number, maxCount: number, isDark: boolean): string {
  if (count === 0) return isDark ? "#1f2937" : "#f9fafb";
  const scale = isDark ? DARK_SCALE : LIGHT_SCALE;
  const idx = Math.min(Math.floor((count / maxCount) * (scale.length - 1)), scale.length - 1);
  return scale[idx];
}

interface Props {
  data: HeatmapData;
}

const CELL_W = 42;
const CELL_H = 28;
const LABEL_LEFT = 160;
const LABEL_TOP = 150;

export function RepoResourceHeatmap({ data }: Props) {
  const navigate = useNavigate();
  const [hover, setHover] = useState<{ repo: string; resource: string; count: number; x: number; y: number } | null>(null);
  const isDark = document.documentElement.classList.contains("dark");

  const maxCount = Math.max(...data.cells.map((c) => c.count), 1);
  const cellMap = new Map<string, number>();
  for (const c of data.cells) cellMap.set(`${c.repo}:${c.resource}`, c.count);

  const svgW = LABEL_LEFT + data.resources.length * CELL_W + 10;
  const svgH = LABEL_TOP + data.repos.length * CELL_H + 10;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">Resource Coverage by Repository</h3>
      <div className="overflow-x-auto relative">
        <svg width={svgW} height={svgH} className="block">
          {data.resources.map((res, ci) => {
            const cx = LABEL_LEFT + ci * CELL_W + CELL_W / 2;
            const cy = LABEL_TOP - 8;
            return (
              <text
                key={res}
                x={cx}
                y={cy}
                textAnchor="start"
                transform={`rotate(-55, ${cx}, ${cy})`}
                fontSize={10}
                fill={isDark ? "#9ca3af" : "#6b7280"}
              >
                {res}
              </text>
            );
          })}
          {data.repos.map((repo, ri) => (
            <text
              key={repo}
              x={LABEL_LEFT - 8}
              y={LABEL_TOP + ri * CELL_H + CELL_H / 2 + 4}
              textAnchor="end"
              fontSize={11}
              fill={isDark ? "#d1d5db" : "#374151"}
            >
              {repo}
            </text>
          ))}
          {data.repos.map((repo, ri) =>
            data.resources.map((res, ci) => {
              const count = cellMap.get(`${repo}:${res}`) ?? 0;
              return (
                <rect
                  key={`${repo}:${res}`}
                  x={LABEL_LEFT + ci * CELL_W}
                  y={LABEL_TOP + ri * CELL_H}
                  width={CELL_W - 1}
                  height={CELL_H - 1}
                  rx={3}
                  fill={getColor(count, maxCount, isDark)}
                  className={count > 0 ? "cursor-pointer" : ""}
                  onClick={() => count > 0 && navigate(`/catalog?repo=${repo}&resource=${res}`)}
                  onMouseEnter={(e) => setHover({ repo, resource: res, count, x: e.clientX, y: e.clientY })}
                  onMouseLeave={() => setHover(null)}
                />
              );
            })
          )}
        </svg>
        {hover && (
          <div
            className="fixed z-50 px-3 py-2 rounded-lg text-xs shadow-lg pointer-events-none"
            style={{
              left: hover.x + 12,
              top: hover.y - 10,
              backgroundColor: "var(--color-tooltip-bg)",
              color: "var(--color-tooltip-text)",
              border: "1px solid var(--color-tooltip-border)",
            }}
          >
            <div className="font-medium">{hover.repo}</div>
            <div>{hover.resource}: {hover.count} tests</div>
          </div>
        )}
      </div>
    </div>
  );
}
