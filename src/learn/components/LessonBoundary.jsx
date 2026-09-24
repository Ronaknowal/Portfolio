import { Component } from "react";

export function LessonLoadError({ onRetry }) {
  return <section className="lesson-load-error" role="alert">
    <h2>This lesson could not be opened</h2>
    <p>Your place in the course is still available. Try again; if the problem persists, reload the page to fetch the current version.</p>
    <div>
      <button type="button" onClick={onRetry}>Try again</button>
      <button type="button" onClick={() => window.location.reload()}>Reload page</button>
    </div>
  </section>;
}

export default class LessonBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) {
    console.error("Lesson rendering failed", error);
    this.props.onError?.();
  }
  render() {
    return this.state.failed
      ? <LessonLoadError onRetry={() => { this.setState({ failed: false }); this.props.onRetry(); }} />
      : this.props.children;
  }
}
