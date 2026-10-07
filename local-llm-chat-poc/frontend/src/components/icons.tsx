import type { ReactNode } from 'react';

function Svg({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

export const Icon = {
  chevron: () => <Svg><path d="M6 9l6 6 6-6" /></Svg>,
  plus: () => <Svg><path d="M12 5v14M5 12h14" /></Svg>,
  up: () => <Svg><path d="M12 19V5M5.5 11.5L12 5l6.5 6.5" /></Svg>,
  down: () => <Svg><path d="M12 5v14M5.5 12.5L12 19l6.5-6.5" /></Svg>,
  stop: () => <Svg><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none" /></Svg>,
  check: () => <Svg><path d="M5 12.5l4.5 4.5L19 7.5" /></Svg>,
  copy: () => <Svg><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></Svg>,
  retry: () => <Svg><path d="M20 11a8 8 0 1 0-2.3 5.7" /><path d="M20 4v7h-7" /></Svg>,
  sliders: () => <Svg><path d="M4 7h8M18 7h2M4 17h2M12 17h8" /><circle cx="15" cy="7" r="2.5" /><circle cx="9" cy="17" r="2.5" /></Svg>,
  close: () => <Svg><path d="M6 6l12 12M18 6L6 18" /></Svg>,
  book: () => <Svg><path d="M5 4.5h9a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3z" /><path d="M5 17a3 3 0 0 1 3-3h9" /></Svg>,
};
