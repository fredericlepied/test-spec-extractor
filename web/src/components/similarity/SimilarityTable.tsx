import type { SimilarityMatch } from "../../types";
import { ScoreBadge } from "./ScoreBadge";

interface Props {
  matches: SimilarityMatch[];
  page: number;
  pageSize: number;
  onPageChange: (p: number) => void;
  sortField: keyof SimilarityMatch;
  sortDir: "asc" | "desc";
  onSort: (field: keyof SimilarityMatch) => void;
  selectedId: number | null;
  onSelect: (id: number) => void;
}

function SortHeader({ label, field, sortField, sortDir, onSort }: {
  label: string;
  field: keyof SimilarityMatch;
  sortField: keyof SimilarityMatch;
  sortDir: "asc" | "desc";
  onSort: (f: keyof SimilarityMatch) => void;
}) {
  return (
    <th
      className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-200 select-none"
      onClick={() => onSort(field)}
    >
      {label} {sortField === field ? (sortDir === "asc" ? "▲" : "▼") : ""}
    </th>
  );
}

export function SimilarityTable({ matches, page, pageSize, onPageChange, sortField, sortDir, onSort, selectedId, onSelect }: Props) {
  const totalPages = Math.ceil(matches.length / pageSize);
  const pageMatches = matches.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div>
      <div className="overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
            <tr>
              <SortHeader label="Test A" field="queryDesc" sortField={sortField} sortDir={sortDir} onSort={onSort} />
              <SortHeader label="Test B" field="matchedDesc" sortField={sortField} sortDir={sortDir} onSort={onSort} />
              <SortHeader label="Score" field="semanticSimilarity" sortField={sortField} sortDir={sortDir} onSort={onSort} />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Type</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
            {pageMatches.map((m) => (
              <tr
                key={m.id}
                onClick={() => onSelect(m.id)}
                className={`cursor-pointer transition-colors ${
                  selectedId === m.id
                    ? "bg-blue-50 dark:bg-blue-900/20"
                    : "hover:bg-gray-50 dark:hover:bg-gray-800/50"
                }`}
              >
                <td className="px-3 py-2 max-w-xs truncate">{m.queryDesc}</td>
                <td className="px-3 py-2 max-w-xs truncate">{m.matchedDesc}</td>
                <td className="px-3 py-2"><ScoreBadge score={m.semanticSimilarity} /></td>
                <td className="px-3 py-2 text-xs text-gray-400">{m.matchType}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
        <span>{matches.length} matches</span>
        <div className="flex gap-2">
          <button disabled={page === 0} onClick={() => onPageChange(page - 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Prev</button>
          <span className="px-2 py-1">{page + 1} / {totalPages}</span>
          <button disabled={page >= totalPages - 1} onClick={() => onPageChange(page + 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Next</button>
        </div>
      </div>
    </div>
  );
}
