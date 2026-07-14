interface Props {
  repos: string[];
  selectedRepos: Set<string>;
  onReposChange: (repos: Set<string>) => void;
  language: "all" | "go" | "python";
  onLanguageChange: (l: "all" | "go" | "python") => void;
  hasTestId: "all" | "yes" | "no";
  onHasTestIdChange: (v: "all" | "yes" | "no") => void;
  resource: string;
  onResourceChange: (r: string) => void;
  availableResources: string[];
}

export function TestFilters({ repos, selectedRepos, onReposChange, language, onLanguageChange, hasTestId, onHasTestIdChange, resource, onResourceChange, availableResources }: Props) {
  const toggleRepo = (repo: string) => {
    const next = new Set(selectedRepos);
    if (next.has(repo)) next.delete(repo);
    else next.add(repo);
    onReposChange(next);
  };

  return (
    <div className="flex flex-wrap gap-6 text-sm">
      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Language</div>
        <div className="flex gap-1">
          {(["all", "go", "python"] as const).map((l) => (
            <button
              key={l}
              onClick={() => onLanguageChange(l)}
              className={`px-2 py-1 rounded text-xs ${
                language === l ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
              }`}
            >
              {l === "all" ? "All" : l === "go" ? "Go" : "Python"}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Polarion ID</div>
        <div className="flex gap-1">
          {(["all", "yes", "no"] as const).map((v) => (
            <button
              key={v}
              onClick={() => onHasTestIdChange(v)}
              className={`px-2 py-1 rounded text-xs ${
                hasTestId === v ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
              }`}
            >
              {v === "all" ? "All" : v === "yes" ? "Has ID" : "No ID"}
            </button>
          ))}
        </div>
      </div>

      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">K8s Resource</div>
        <select
          value={resource}
          onChange={(e) => onResourceChange(e.target.value)}
          className="px-2 py-1 rounded text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 border-none"
        >
          <option value="">All</option>
          {availableResources.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      <div>
        <div className="font-medium text-gray-500 dark:text-gray-400 mb-2">Repositories</div>
        <div className="flex flex-wrap gap-2">
          {repos.map((repo) => (
            <label key={repo} className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={selectedRepos.has(repo)} onChange={() => toggleRepo(repo)} className="rounded" />
              <span className="text-xs text-gray-600 dark:text-gray-300">{repo}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
