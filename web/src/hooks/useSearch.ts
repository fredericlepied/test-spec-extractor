import { useState, useMemo, useCallback, useRef, useEffect } from "react";

export function useSearch<T>(
  items: T[],
  searchFields: (item: T) => string,
  delay = 300,
  initialQuery = ""
): {
  query: string;
  setQuery: (q: string) => void;
  results: T[];
} {
  const [query, setQueryRaw] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const setQuery = useCallback(
    (q: string) => {
      setQueryRaw(q);
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setDebouncedQuery(q), delay);
    },
    [delay]
  );

  useEffect(() => {
    return () => clearTimeout(timerRef.current);
  }, []);

  const searchIndex = useMemo(
    () => items.map((item) => ({ item, text: searchFields(item).toLowerCase() })),
    [items, searchFields]
  );

  const results = useMemo(() => {
    if (!debouncedQuery.trim()) return items;
    const terms = debouncedQuery.toLowerCase().split(/\s+/);
    return searchIndex
      .filter(({ text }) => terms.every((t) => text.includes(t)))
      .map(({ item }) => item);
  }, [searchIndex, debouncedQuery, items]);

  return { query, setQuery, results };
}
