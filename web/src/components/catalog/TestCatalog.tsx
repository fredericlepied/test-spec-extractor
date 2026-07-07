import { useState, useMemo, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { useData } from "../../hooks/useData";
import { useSearch } from "../../hooks/useSearch";
import type { TestRecord } from "../../types";
import { TestFilters } from "./TestFilters";
import { TestTable } from "./TestTable";

const VALID_LANGS = ["all", "go", "python"] as const;
type Lang = (typeof VALID_LANGS)[number];

export function TestCatalog() {
  const [searchParams] = useSearchParams();
  const { data: tests, loading, error } = useData<TestRecord[]>("/data/tests.json");

  const initialLang = VALID_LANGS.includes(searchParams.get("lang") as Lang)
    ? (searchParams.get("lang") as Lang)
    : "all";
  const initialRepo = searchParams.get("repo") || "";
  const initialResource = searchParams.get("resource") || "";
  const initialSearch = searchParams.get("search") || "";

  const [selectedRepos, setSelectedRepos] = useState<Set<string>>(new Set());
  const [language, setLanguage] = useState<Lang>(initialLang);
  const [hasTestId, setHasTestId] = useState<"all" | "yes" | "no">("all");
  const [resource, setResource] = useState(initialResource);
  const [page, setPage] = useState(0);

  const repos = useMemo(() => {
    if (!tests) return [];
    return [...new Set(tests.map((t) => t.repo))].sort();
  }, [tests]);

  useMemo(() => {
    if (repos.length > 0 && selectedRepos.size === 0) {
      if (initialRepo && repos.includes(initialRepo)) {
        setSelectedRepos(new Set([initialRepo]));
      } else {
        setSelectedRepos(new Set(repos));
      }
    }
  }, [repos, selectedRepos.size, initialRepo]);

  const availableResources = useMemo(() => {
    if (!tests) return [];
    const counts = new Map<string, number>();
    for (const t of tests) {
      for (const r of t.k8sResources) counts.set(r, (counts.get(r) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([r]) => r);
  }, [tests]);

  const preFiltered = useMemo(() => {
    if (!tests) return [];
    return tests
      .filter((t) => selectedRepos.size === 0 || selectedRepos.has(t.repo))
      .filter((t) => language === "all" || t.language === language)
      .filter((t) => hasTestId === "all" || (hasTestId === "yes" ? t.testId : !t.testId))
      .filter((t) => !resource || t.k8sResources.includes(resource));
  }, [tests, selectedRepos, language, hasTestId, resource]);

  const searchFields = useCallback(
    (t: TestRecord) => [t.desc, ...t.steps, ...t.validations].join(" "),
    []
  );
  const { query, setQuery, results } = useSearch(preFiltered, searchFields, 300, initialSearch);

  if (loading) return <div className="text-gray-400">Loading test catalog...</div>;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  return (
    <div className="space-y-4">
      <input
        type="text"
        value={query}
        onChange={(e) => { setQuery(e.target.value); setPage(0); }}
        placeholder="Search tests..."
        className="w-full px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <TestFilters
        repos={repos}
        selectedRepos={selectedRepos}
        onReposChange={(r) => { setSelectedRepos(r); setPage(0); }}
        language={language}
        onLanguageChange={(l) => { setLanguage(l); setPage(0); }}
        hasTestId={hasTestId}
        onHasTestIdChange={(v) => { setHasTestId(v); setPage(0); }}
        resource={resource}
        onResourceChange={(r) => { setResource(r); setPage(0); }}
        availableResources={availableResources}
      />
      <TestTable tests={results} page={page} pageSize={50} onPageChange={setPage} />
    </div>
  );
}
