import { useMemo, useState } from "react";
import { useData } from "../../hooks/useData";
import type { DuplicateCluster, ClusterTest, ClusterEdge } from "../../types";
import { ForceGraph } from "./ForceGraph";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

export function SimilarityGraph() {
  const { data: clusters, loading, error } = useData<DuplicateCluster[]>("/data/clusters.json");
  const [hiddenRepos, setHiddenRepos] = useState<Set<string>>(new Set());
  const [minSize, setMinSize] = useState(2);

  const repos = useMemo(() => {
    if (!clusters) return [];
    const s = new Map<string, number>();
    for (const c of clusters)
      for (const t of c.tests)
        if (!s.has(t.repo)) s.set(t.repo, t.colorIndex);
    return [...s.entries()].map(([name, colorIndex]) => ({ name, colorIndex }));
  }, [clusters]);

  const { nodes, edges } = useMemo(() => {
    if (!clusters) return { nodes: [] as ClusterTest[], edges: [] as ClusterEdge[] };
    const allNodes: ClusterTest[] = [];
    const allEdges: ClusterEdge[] = [];
    for (const c of clusters) {
      if (c.size < minSize) continue;
      const visibleTests = c.tests.filter((t) => !hiddenRepos.has(t.repo));
      if (visibleTests.length < 2) continue;
      const idxMap = new Map<number, number>();
      c.tests.forEach((t, i) => {
        if (!hiddenRepos.has(t.repo)) {
          idxMap.set(i, allNodes.length);
          allNodes.push(t);
        }
      });
      for (const e of c.edges) {
        const si = idxMap.get(e.sourceIdx);
        const ti = idxMap.get(e.targetIdx);
        if (si !== undefined && ti !== undefined) {
          allEdges.push({ sourceIdx: si, targetIdx: ti, score: e.score });
        }
      }
    }
    return { nodes: allNodes, edges: allEdges };
  }, [clusters, hiddenRepos, minSize]);

  const toggleRepo = (repo: string) => {
    setHiddenRepos((prev) => {
      const next = new Set(prev);
      if (next.has(repo)) next.delete(repo);
      else next.add(repo);
      return next;
    });
  };

  if (loading) return <div className="text-gray-400">Loading graph...</div>;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  return (
    <div className="flex gap-4 h-full">
      <div className="w-44 shrink-0 space-y-4 text-sm">
        <div>
          <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Repositories</div>
          <div className="space-y-1">
            {repos.map((r) => (
              <label key={r.name} className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={!hiddenRepos.has(r.name)} onChange={() => toggleRepo(r.name)} className="rounded" />
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: COLORS[r.colorIndex] }} />
                <span className="text-xs text-gray-600 dark:text-gray-300 truncate">{r.name}</span>
              </label>
            ))}
          </div>
        </div>
        <div>
          <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Min cluster size</div>
          <select value={minSize} onChange={(e) => setMinSize(Number(e.target.value))} className="px-2 py-1 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
            {[2, 3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="text-xs text-gray-400 space-y-2">
          <div>{nodes.length} nodes, {edges.length} edges</div>
          <div className="border-t border-gray-200 dark:border-gray-700 pt-2">
            <div className="font-medium text-gray-500 dark:text-gray-400 mb-1">How to read this</div>
            <p>Each dot is a test. Lines connect tests with similarity &ge; 90%. Clusters of connected dots are groups of near-duplicates.</p>
            <p className="mt-1">Dot size = number of similar tests. Color = repository.</p>
          </div>
          <div className="border-t border-gray-200 dark:border-gray-700 pt-2">
            <div className="font-medium text-gray-500 dark:text-gray-400 mb-1">Controls</div>
            <p>Scroll to zoom. Drag to pan. Hover for details. Click a dot to find it in the catalog.</p>
          </div>
        </div>
      </div>
      <div className="flex-1 min-w-0 h-full">
        <ForceGraph nodes={nodes} edges={edges} />
      </div>
    </div>
  );
}
