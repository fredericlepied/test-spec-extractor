interface Props {
  repos: string[];
  selectedRepos: Set<string>;
  onReposChange: (repos: Set<string>) => void;
  matchType: "all" | "go-go" | "py-py" | "cross";
  onMatchTypeChange: (t: "all" | "go-go" | "py-py" | "cross") => void;
  threshold: number;
  onThresholdChange: (t: number) => void;
  crossRepoOnly: boolean;
  onCrossRepoOnlyChange: (v: boolean) => void;
}

export function SimilarityFilters({ repos, selectedRepos, onReposChange, matchType, onMatchTypeChange, threshold, onThresholdChange, crossRepoOnly, onCrossRepoOnlyChange }: Props) {
  const toggleRepo = (repo: string) => {
    const next = new Set(selectedRepos);
    if (next.has(repo)) next.delete(repo);
    else next.add(repo);
    onReposChange(next);
  };

  return (
    <div className="space-y-4 text-sm">
      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Match Type</div>
        <div className="flex flex-wrap gap-1">
          {(["all", "go-go", "py-py", "cross"] as const).map((t) => (
            <button
              key={t}
              onClick={() => onMatchTypeChange(t)}
              className={`px-2 py-1 rounded text-xs ${
                matchType === t
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
              }`}
            >
              {t === "all" ? "All" : t}
            </button>
          ))}
        </div>
      </div>

      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={crossRepoOnly}
          onChange={(e) => onCrossRepoOnlyChange(e.target.checked)}
          className="rounded"
        />
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Cross-repo only</span>
      </label>

      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">
          Min Score: {(threshold * 100).toFixed(0)}%
        </div>
        <input
          type="range"
          min={0.65}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Repositories</div>
        <div className="space-y-1">
          {repos.map((repo) => (
            <label key={repo} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={selectedRepos.has(repo)}
                onChange={() => toggleRepo(repo)}
                className="rounded"
              />
              <span className="text-xs text-gray-600 dark:text-gray-300 truncate">{repo}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
