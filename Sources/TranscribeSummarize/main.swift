// ABOUTME: Root command for the transcribe CLI with subcommand dispatch.
// ABOUTME: Handles backward compatibility for deprecated transcribe-summarize invocation.

import ArgumentParser
import Foundation

@main
struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: HelpText.text(.rootAbstract),
        discussion: HelpText.text(.rootDiscussion),
        version: "0.2.21",
        subcommands: [SummarizeCommand.self, TextCommand.self, SRTCommand.self, VTTCommand.self, WordsCommand.self]
    )

    /// Override the default entry point to handle backward compatibility.
    /// When invoked as `transcribe-summarize`, auto-inject the `summarize` subcommand
    /// and print a deprecation warning.
    static func main() async {
        let invocationName = URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent

        if invocationName == "transcribe-summarize" {
            fputs("Warning: 'transcribe-summarize' is deprecated. Use 'transcribe summarize' instead.\n", stderr)
            let args = legacyArguments(Array(CommandLine.arguments.dropFirst()))

            do {
                var command = try parseAsRoot(args)
                if var asyncCmd = command as? AsyncParsableCommand {
                    try await asyncCmd.run()
                } else {
                    try command.run()
                }
            } catch {
                exit(withError: error)
            }
        } else {
            do {
                var command = try parseAsRoot()
                if var asyncCmd = command as? AsyncParsableCommand {
                    try await asyncCmd.run()
                } else {
                    try command.run()
                }
            } catch {
                exit(withError: error)
            }
        }
    }

    static func legacyArguments(_ arguments: [String]) -> [String] {
        let rootArguments: Set<String> = ["--help", "-h", "--version"]
        let knownSubcommands: Set<String> = ["summarize", "text", "srt", "vtt", "words", "help"]

        guard let first = arguments.first else {
            return ["summarize"]
        }
        if rootArguments.contains(first) || knownSubcommands.contains(first) {
            return arguments
        }
        return ["summarize"] + arguments
    }
}
