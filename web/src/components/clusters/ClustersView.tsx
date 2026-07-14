import { useState, useMemo } from "react";
import { useData } from "../../hooks/useData";
import type { DuplicateCluster } from "../../types";
import { ClusterCard } from "./ClusterCard";

export function ClustersView() {
  const { data: clusters, loading, error } = useData<DuplicateCluster[]>("/data/clusters.json");

  const [crossRepoOnly, setCrossRepoOnly] = useState(false);
  const [minSize, setMinSize] = useState(2);
  const [sortBy, setSortBy] = useState<"size" | "score" | "repos">("size");
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const filtered = useMemo(() => {
    if (!clusters) return [];
    let result = clusters
      .filter((c) => c.size >= minSize)
      .filter((c) => !crossRepoOnly || c.isCrossRepo);

    if (sortBy === "score") result = [...result].sort((a, b) => b.maxScore - a.maxScore);
    else if (sortBy === "repos") result = [...result].sort((a, b) => b.repos.length - a.repos.length || b.size - a.size);

    return result;
  }, [clusters, crossRepoOnly, minSize, sortBy]);

  const totalPages = Math.ceil(filtered.length / pageSize);
  const pageItems = filtered.slice(page * pageSize, (page + 1) * pageSize);

  const totalTests = useMemo(() => filtered.reduce((s, c) => s + c.size, 0), [filtered]);
  const crossCount = useMemo(() => filtered.filter((c) => c.isCrossRepo).length, [filtered]);

  if (loading) return <div className="text-gray-400">Loading clusters...</div>;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <div className="text-gray-500 dark:text-gray-400">
          {filtered.length} clusters, {totalTests} tests, {crossCount} cross-repo
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={crossRepoOnly} onChange={(e) => { setCrossRepoOnly(e.target.checked); setPage(0); }} className="rounded" />
          <span className="text-xs text-gray-500 dark:text-gray-400">Cross-repo only</span>
        </label>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-400">Min size:</span>
          <select value={minSize} onChange={(e) => { setMinSize(Number(e.target.value)); setPage(0); }} className="px-1 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
            {[2, 3, 5, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-500 dark:text-gray-400">Sort:</span>
          <select value={sortBy} onChange={(e) => { setSortBy(e.target.value as "size" | "score" | "repos"); setPage(0); }} className="px-1 py-0.5 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
            <option value="size">Size</option>
            <option value="score">Score</option>
            <option value="repos">Repos</option>
          </select>
        </div>
      </div>

      <div className="space-y-3">
        {pageItems.map((c) => <ClusterCard key={c.id} cluster={c} />)}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
          <button disabled={page === 0} onClick={() => setPage(page - 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Prev</button>
          <span>{page + 1} / {totalPages}</span>
          <button disabled={page >= totalPages - 1} onClick={() => setPage(page + 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Next</button>
        </div>
      )}
    </div>
  );
}
