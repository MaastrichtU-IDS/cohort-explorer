type PageExitTarget = Pick<EventTarget, 'addEventListener' | 'removeEventListener'>;

export function abortOnPageExit(
  controller: AbortController,
  target: PageExitTarget = window,
  onPersistedRestore?: () => void
): () => void {
  const abort = () => controller.abort();
  const restore = (event: Event) => {
    if ((event as PageTransitionEvent).persisted && controller.signal.aborted) {
      onPersistedRestore?.();
    }
  };
  target.addEventListener('pagehide', abort, {once: true});
  target.addEventListener('pageshow', restore);
  return () => {
    target.removeEventListener('pagehide', abort);
    target.removeEventListener('pageshow', restore);
  };
}
