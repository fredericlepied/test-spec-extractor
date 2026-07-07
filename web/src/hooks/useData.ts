import { useState, useEffect } from "react";

const cache = new Map<string, unknown>();

export function useData<T>(url: string): {
  data: T | null;
  loading: boolean;
  error: Error | null;
} {
  const [data, setData] = useState<T | null>(
    (cache.get(url) as T) ?? null
  );
  const [loading, setLoading] = useState(!cache.has(url));
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (cache.has(url)) return;

    let cancelled = false;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((d) => {
        if (cancelled) return;
        cache.set(url, d);
        setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  return { data, loading, error };
}
