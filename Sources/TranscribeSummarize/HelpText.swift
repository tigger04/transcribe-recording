// ABOUTME: Loads canonical CLI help text from the project documentation.
// ABOUTME: Resolves source-tree and installed shared-data locations at runtime.

import ArgumentParser
import Foundation

enum HelpText {
    enum Key: String, CaseIterable {
        case rootAbstract = "root.abstract"
        case rootDiscussion = "root.discussion"
        case commonInputFile = "common.input-file"
        case commonOutput = "common.output"
        case commonModel = "common.model"
        case commonSpeakers = "common.speakers"
        case commonPreprocess = "common.preprocess"
        case commonDevice = "common.device"
        case commonVerbose = "common.verbose"
        case summarizeAbstract = "summarize.abstract"
        case summarizeDiscussion = "summarize.discussion"
        case summarizeFormat = "summarize.format"
        case summarizeTimestamps = "summarize.timestamps"
        case summarizeConfidence = "summarize.confidence"
        case summarizeLLM = "summarize.llm"
        case summarizeDryRun = "summarize.dry-run"
        case textAbstract = "text.abstract"
        case textFormat = "text.format"
        case textTimestamps = "text.timestamps"
        case srtAbstract = "srt.abstract"
        case subtitleMaxLength = "subtitle.max-len"
        case subtitleSplitOnWord = "subtitle.split-on-word"
        case vttAbstract = "vtt.abstract"
        case wordsAbstract = "words.abstract"
    }

    enum LoadError: LocalizedError {
        case documentNotFound([URL])

        var errorDescription: String? {
            switch self {
            case .documentNotFound(let urls):
                let paths = urls.map(\.path).joined(separator: ", ")
                return "Canonical CLI help document not found. Searched: \(paths)"
            }
        }
    }

    private static let sections: [String: String] = {
        do {
            return parse(try load(from: defaultURLs()))
        } catch {
            fatalError(error.localizedDescription)
        }
    }()

    static func text(_ key: Key) -> String {
        guard let value = sections[key.rawValue] else {
            fatalError("Missing CLI help section: \(key.rawValue)")
        }
        return value
    }

    static func argument(_ key: Key) -> ArgumentHelp {
        ArgumentHelp(text(key))
    }

    static func load(from urls: [URL]) throws -> String {
        for url in urls where FileManager.default.fileExists(atPath: url.path) {
            return try String(contentsOf: url, encoding: .utf8)
        }
        throw LoadError.documentNotFound(urls)
    }

    static func parse(_ document: String) -> [String: String] {
        var result: [String: String] = [:]
        var currentKey: String?
        var currentLines: [String] = []

        func storeCurrentSection() {
            guard let currentKey else { return }
            let value = currentLines.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            result[currentKey] = value
        }

        for line in document.components(separatedBy: .newlines) {
            if line.hasPrefix("## ") {
                storeCurrentSection()
                currentKey = String(line.dropFirst(3))
                currentLines = []
            } else if currentKey != nil {
                currentLines.append(line)
            }
        }
        storeCurrentSection()
        return result
    }

    private static func defaultURLs() -> [URL] {
        let fileManager = FileManager.default
        let fileName = "transcribe-help.md"
        var urls = [
            URL(fileURLWithPath: fileManager.currentDirectoryPath)
                .appendingPathComponent("docs")
                .appendingPathComponent(fileName)
        ]

        let invokedURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let binaryDirectory = invokedURL.deletingLastPathComponent()
        urls.append(
            binaryDirectory.deletingLastPathComponent()
                .appendingPathComponent("share/transcribe-summarize")
                .appendingPathComponent(fileName)
        )

        var ancestor = invokedURL.resolvingSymlinksInPath().deletingLastPathComponent()
        for _ in 0..<6 {
            urls.append(ancestor.appendingPathComponent("docs").appendingPathComponent(fileName))
            ancestor.deleteLastPathComponent()
        }

        urls.append(
            fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent(".local/share/transcribe-summarize")
                .appendingPathComponent(fileName)
        )
        return urls
    }
}
