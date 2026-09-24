// Compact curriculum briefs, not manuscripts or completed content checkpoints.
// Every scope step, investigation and exercise is authored for its topic. The
// helper only maps those fields into the existing planned-page schema.
export function professionalTopic(title, level, prerequisites, sequence, visual, practice, misconception, sources) {
  return {
    title,
    level,
    blueprint: {
      summary: `${sequence[0]}. ${sequence[1]}.`,
      prerequisites,
      outcomes: [sequence[1], sequence[sequence.length - 1]],
      sequence,
      visual: { type: visual[0], question: visual[1], interaction: visual[2] },
      practice: { task: practice[0], success: practice[1] },
      misconceptions: [misconception],
      sources,
      depth: level === "advanced" ? "specialist" : level === "frontier" ? "frontier" : "core",
      reviewFocus: "Starting curriculum brief, not completed lesson research. Verify specific claims against dated primary sources; expand the concept-level design, worked examples, independent practice and topic-specific visuals before authoring. Use multiple investigations when the mechanisms require them.",
    },
  };
}
