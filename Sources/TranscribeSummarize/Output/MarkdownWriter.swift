// ABOUTME: Assembles final markdown output matching vision spec format.
// ABOUTME: Combines summary, metadata, and transcript into structured document.

import Foundation

struct MarkdownWriter {
    private let confidenceThreshold: Double
    private let includeTimestamps: Bool
    private let includeLowConfidence: Bool

    init(
        confidenceThreshold: Double = 0.8,
        includeTimestamps: Bool = true,
        includeLowConfidence: Bool = true
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.includeTimestamps = includeTimestamps
        self.includeLowConfidence = includeLowConfidence
    }

    func write(
        summary: Summary,
        segments: [Segment],
        to outputPath: String
    ) throws {
        var output = ""

        output += "# \(summary.title)\n\n"

        if let date = summary.date {
            output += "**Date:** \(date)  \n"
        }
        output += "**Duration:** \(summary.duration)  \n"
        if !summary.participants.isEmpty {
            output += "**Participants:** \(summary.participants.joined(separator: ", "))  \n"
        }
        output += "**Transcription Confidence:** \(summary.confidenceRating)\n\n"

        output += "## Summary\n\n"

        if let agenda = summary.agenda, !agenda.isEmpty {
            output += "### Agenda\n\n\(agenda)\n\n"
        }

        output += "### Key Points\n\n"

        if !summary.keyPoints.decisions.isEmpty {
            output += "**Decisions:**\n"
            for decision in summary.keyPoints.decisions {
                output += "- \(decision)\n"
            }
            output += "\n"
        }

        if !summary.keyPoints.discussions.isEmpty {
            output += "**Significant Discussions:**\n"
            for discussion in summary.keyPoints.discussions {
                output += "- \(discussion)\n"
            }
            output += "\n"
        }

        if !summary.keyPoints.unresolvedQuestions.isEmpty {
            output += "**Unresolved Questions:**\n"
            for question in summary.keyPoints.unresolvedQuestions {
                output += "- \(question)\n"
            }
            output += "\n"
        }

        output += "### Themes and Tone\n\n\(summary.themesAndTone)\n\n"

        output += "### Actions\n\n"
        if summary.actions.isEmpty {
            output += "*No action items identified.*\n\n"
        } else {
            output += "| Action | Assigned To | Due Date |\n"
            output += "|--------|-------------|----------|\n"
            for action in summary.actions {
                let assignee = action.assignedTo ?? "—"
                let due = action.dueDate ?? "—"
                output += "| \(action.description) | \(assignee) | \(due) |\n"
            }
            output += "\n"
        }

        output += "### Conclusion\n\n\(summary.conclusion)\n\n"

        output += "## Transcript\n\n"
        output += writeTranscript(segments: segments)

        try output.write(toFile: outputPath, atomically: true, encoding: .utf8)
    }

    private func writeTranscript(segments: [Segment]) -> String {
        var output = ""

        for segment in segments {
            if !includeLowConfidence && segment.confidence < confidenceThreshold {
                continue
            }

            var line = ""

            if segment.confidence < confidenceThreshold {
                line += "[Confidence: low] "
            }

            if includeTimestamps {
                line += "[\(segment.startTimestamp)] "
            }

            if let speaker = segment.speaker {
                line += "**\(speaker):** "
            }

            line += segment.text

            output += line + "  \n"
        }

        return output
    }

    static func calculateConfidenceRating(segments: [Segment]) -> String {
        guard !segments.isEmpty else { return "N/A" }

        let totalConfidence = segments.reduce(0.0) { $0 + $1.confidence }
        let average = totalConfidence / Double(segments.count)
        let percentage = Int(average * 100)

        let rating: String
        switch percentage {
        case 95...100: rating = "Excellent"
        case 85..<95: rating = "Good"
        case 70..<85: rating = "Fair"
        default: rating = "Poor"
        }

        return "\(percentage)% (\(rating))"
    }

    static func formatDuration(_ seconds: Double) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60

        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        } else {
            return String(format: "%d:%02d", minutes, secs)
        }
    }

    static func defaultTitle(from inputPath: String) -> String {
        let url = URL(fileURLWithPath: inputPath)
        let filename = url.deletingPathExtension().lastPathComponent

        return filename
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .capitalized
    }
}
