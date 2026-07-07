import { useState } from "react";
import type { DuplicateCluster } from "../../types";
import { ScoreBadge } from "../similarity/ScoreBadge";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];
const POLARION_URL = "https://polarion.engineering.redhat.com/polarion/#/project/OSE/workitem?id=OCP-";

interface Props {
  cluster: DuplicateCluster;
}

export function ClusterCard({ cluster }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
      <div
        className="flex items-center gap-3 p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-xs text-gray-400 w-8">#{cluster.id + 1}</span>
        <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs font-medium">
          {cluster.size} tests
        </span>
        <ScoreBadge score={cluster.maxScore} />
        <div className="flex gap-1">
          {cluster.repos.map((repo) => (
            <span key={repo} className="px-1.5 py-0.5 rounded text-[10px] text-white" style={{ backgroundColor: COLORS[cluster.tests.find((t) => t.repo === repo)?.colorIndex ?? 0] }}>
              {repo}
            </span>
          ))}
        </div>
        {cluster.isCrossRepo && (
          <span className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded text-[10px]">cross-repo</span>
        )}
        <span className="ml-auto text-xs text-gray-400">{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="border-t border-gray-100 dark:border-gray-700 p-4 space-y-2">
          <div className="text-xs text-gray-400 mb-2">{cluster.edges.length} pairs, avg score {(cluster.avgScore * 100).toFixed(1)}%</div>
          {cluster.tests.map((t, i) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <span className="w-2 h-2 mt-1.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[t.colorIndex] }} />
              <div className="min-w-0">
                <div className="truncate">{t.desc}</div>
                <div className="flex gap-2 text-xs text-gray-400">
                  <span>{t.repo}</span>
                  {t.sourceUrl ? (
                    <a href={t.sourceUrl} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline truncate">{t.file}</a>
                  ) : (
                    <span className="truncate">{t.file}</span>
                  )}
                  {t.testId && (
                    <a href={`${POLARION_URL}${t.testId}`} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">OCP-{t.testId}</a>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
