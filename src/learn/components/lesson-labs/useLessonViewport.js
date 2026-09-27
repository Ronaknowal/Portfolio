import { useEffect, useRef, useState } from 'react';

// Keep large investigations deferred until their visible section is nearby.
export default function useLessonViewport() {
  const container = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!globalThis.IntersectionObserver) {
      setReady(true);
      return undefined;
    }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setReady(true);
        observer.disconnect();
      }
    }, { rootMargin: '240px' });
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);

  return [container, ready];
}
