import { useNavigate } from "react-router-dom";
import { useData } from "../../hooks/useData";
import type { DashboardStats } from "../../types";
import { StatCard } from "./StatCard";
import { RepoBarChart } from "./RepoBarChart";
import { ScoreHistogram } from "./ScoreHistogram";
import { K8sResourceChart } from "./K8sResourceChart";
import { RepoResourceHeatmap } from "./RepoResourceHeatmap";

export function Dashboard() {
  const { data: stats, loading, error } = useData<DashboardStats>("/data/stats.json");

  const navigate = useNavigate();

  if (loading) return <div className="text-gray-400">Loading dashboard...</div>;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;
  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Tests" value={stats.totalTests} detail={`${stats.goTests} Go / ${stats.pyTests} Python`} to="/catalog" />
        <StatCard label="Similar Pairs" value={stats.totalMatches} detail="Each test can appear in multiple pairs" to="/similarity" />
        <StatCard label="Cross-Language Pairs" value={stats.crossLanguageMatches} detail={`${stats.matchTypes.goGo} Go-Go / ${stats.matchTypes.pyPy} Py-Py`} to="/similarity?type=cross" />
        <StatCard label="Avg Similarity" value={stats.avgSimilarity.toFixed(3)} detail={`${stats.testIdCoverage.withId} with Polarion ID`} to="/similarity?min=0.90" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RepoBarChart data={stats.repos} />
        <ScoreHistogram data={stats.scoreDistribution} />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
          Test Overlap per Repository <span className="font-normal">(near-duplicates, score &ge; 0.90)</span>
        </h3>
        <div className="flex gap-4 mb-4 text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block" /> Cross-repo</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-500 inline-block" /> Internal</span>
        </div>
        <div className="space-y-3">
          {stats.repoOverlap.map((r) => {
            const totalPct = r.crossRepoPct + r.internalPct;
            return (
              <div key={r.repo} className="flex items-center gap-3 text-sm">
                <div className="w-40 truncate text-gray-600 dark:text-gray-300">{r.repo}</div>
                <div className="flex-1 bg-gray-100 dark:bg-gray-700 rounded-full h-5 overflow-hidden flex">
                  {r.crossRepoPct > 0 && (
                    <div
                      className="h-full bg-blue-500 cursor-pointer hover:bg-blue-600 transition-colors"
                      style={{ width: `${r.crossRepoPct}%` }}
                      title={`Cross-repo: ${r.crossRepo} tests (${r.crossRepoPct}%)`}
                      onClick={() => navigate(`/similarity?repo=${r.repo}&crossRepo=1&min=0.90`)}
                    />
                  )}
                  {r.internalPct > 0 && (
                    <div
                      className="h-full bg-amber-500 cursor-pointer hover:bg-amber-600 transition-colors"
                      style={{ width: `${r.internalPct}%` }}
                      title={`Internal: ${r.internal} tests (${r.internalPct}%)`}
                      onClick={() => navigate(`/similarity?repo=${r.repo}&sameRepo=1&min=0.90`)}
                    />
                  )}
                </div>
                <div className="w-36 text-xs text-gray-400 text-right">
                  {totalPct > 0 ? `${totalPct.toFixed(1)}%` : "0%"}
                  {" "}({r.crossRepo}+{r.internal}/{r.total})
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <K8sResourceChart data={stats.k8sResources} />
      <RepoResourceHeatmap data={stats.heatmap} />
      {stats.commonK8sResources.length > 0 && (
        <div className="text-xs text-gray-400 dark:text-gray-500">
          Filtered common resources (&gt;30% frequency): {stats.commonK8sResources.join(", ")}
        </div>
      )}
    </div>
  );
}
