import { useState, useMemo, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useData } from "../../hooks/useData";
import type { SimilarityMatch, TestRecord } from "../../types";
import { SimilarityFilters } from "./SimilarityFilters";
import { SimilarityTable } from "./SimilarityTable";
import { SimilarityDetail } from "./SimilarityDetail";

const VALID_TYPES = ["all", "go-go", "py-py", "cross"] as const;
type MatchType = (typeof VALID_TYPES)[number];

export function SimilarityExplorer() {
  const [searchParams] = useSearchParams();
  const { data: matches, loading, error } = useData<SimilarityMatch[]>("/data/similarity.json");
  const { data: tests } = useData<TestRecord[]>("/data/tests.json");

  const initialType = VALID_TYPES.includes(searchParams.get("type") as MatchType)
    ? (searchParams.get("type") as MatchType)
    : "all";
  const initialThreshold = searchParams.has("min")
    ? parseFloat(searchParams.get("min")!)
    : 0.65;
  const initialRepo = searchParams.get("repo") || "";
  const initialCrossRepo = searchParams.get("crossRepo") === "1";
  const initialSameRepo = searchParams.get("sameRepo") === "1";

  const [selectedRepos, setSelectedRepos] = useState<Set<string>>(new Set());
  const [matchType, setMatchType] = useState<MatchType>(initialType);
  const [threshold, setThreshold] = useState(isNaN(initialThreshold) ? 0.65 : initialThreshold);
  const [page, setPage] = useState(0);
  const [sortField, setSortField] = useState<keyof SimilarityMatch>("semanticSimilarity");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [crossRepoOnly, setCrossRepoOnly] = useState(initialCrossRepo);
  const [sameRepoOnly] = useState(initialSameRepo);

  const repos = useMemo(() => {
    if (!matches) return [];
    const s = new Set<string>();
    matches.forEach((m) => { s.add(m.queryRepo); s.add(m.matchedRepo); });
    return [...s].sort();
  }, [matches]);

  useMemo(() => {
    if (repos.length > 0 && selectedRepos.size === 0) {
      if (initialRepo && repos.includes(initialRepo)) {
        setSelectedRepos(new Set([initialRepo]));
      } else {
        setSelectedRepos(new Set(repos));
      }
    }
  }, [repos, selectedRepos.size, initialRepo]);

  const filtered = useMemo(() => {
    if (!matches) return [];
    return matches
      .filter((m) => m.semanticSimilarity >= threshold)
      .filter((m) => matchType === "all" || m.matchType === matchType)
      .filter((m) => selectedRepos.size === 0 || selectedRepos.has(m.queryRepo) || selectedRepos.has(m.matchedRepo))
      .filter((m) => !crossRepoOnly || m.queryRepo !== m.matchedRepo)
      .filter((m) => !sameRepoOnly || m.queryRepo === m.matchedRepo)
      .sort((a, b) => {
        const av = a[sortField], bv = b[sortField];
        if (typeof av === "number" && typeof bv === "number") return sortDir === "asc" ? av - bv : bv - av;
        return sortDir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
  }, [matches, threshold, matchType, selectedRepos, sortField, sortDir, crossRepoOnly, sameRepoOnly]);

  const handleSort = (field: keyof SimilarityMatch) => {
    if (field === sortField) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortField(field); setSortDir("desc"); }
    setPage(0);
  };

  const testIndex = useMemo(() => {
    if (!tests) return new Map<string, TestRecord>();
    const m = new Map<string, TestRecord>();
    for (const t of tests) {
      m.set(`${t.repo}:${t.desc}`, t);
    }
    return m;
  }, [tests]);

  const selectedMatch = matches?.find((m) => m.id === selectedId) ?? null;
  const queryTest = selectedMatch ? testIndex.get(`${selectedMatch.queryRepo}:${selectedMatch.queryDesc}`) ?? null : null;
  const matchedTest = selectedMatch ? testIndex.get(`${selectedMatch.matchedRepo}:${selectedMatch.matchedDesc}`) ?? null : null;
  const detailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedMatch && detailRef.current) {
      detailRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedMatch]);

  if (loading) return <div className="text-gray-400">Loading similarity data...</div>;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  return (
    <div className="flex gap-6 h-full">
      <div className="w-48 shrink-0">
        <SimilarityFilters
          repos={repos}
          selectedRepos={selectedRepos}
          onReposChange={(r) => { setSelectedRepos(r); setPage(0); }}
          matchType={matchType}
          onMatchTypeChange={(t) => { setMatchType(t); setPage(0); }}
          threshold={threshold}
          onThresholdChange={(t) => { setThreshold(t); setPage(0); }}
          crossRepoOnly={crossRepoOnly}
          onCrossRepoOnlyChange={(v) => { setCrossRepoOnly(v); setPage(0); }}
        />
      </div>
      <div className="flex-1 min-w-0 flex flex-col gap-4 overflow-hidden">
        {selectedMatch && (
          <div className="shrink-0 max-h-[40vh] overflow-auto">
            <SimilarityDetail match={selectedMatch} queryTest={queryTest} matchedTest={matchedTest} />
          </div>
        )}
        <div className="flex-1 overflow-auto min-h-0">
          <SimilarityTable
            matches={filtered}
            page={page}
            pageSize={50}
            onPageChange={setPage}
            sortField={sortField}
            sortDir={sortDir}
            onSort={handleSort}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>
      </div>
    </div>
  );
}
