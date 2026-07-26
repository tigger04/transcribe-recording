// ABOUTME: Tests that root CLI help is loaded from the canonical documentation file.
// ABOUTME: Guards the words guidance and runtime help-file discovery contract.

import ArgumentParser
import Foundation
import XCTest

@testable import TranscribeSummarize

final class HelpTextTests: XCTestCase {
    func testLoadUsesFirstAvailableHelpDocument_RT37_1() throws {
        let testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: testDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: testDirectory) }

        let missingURL = testDirectory.appendingPathComponent("missing.md")
        let helpURL = testDirectory.appendingPathComponent("transcribe-help.md")
        let expected = "Canonical help text\n"
        try expected.write(to: helpURL, atomically: true, encoding: .utf8)

        XCTAssertEqual(try HelpText.load(from: [missingURL, helpURL]), expected)
    }

    func testRootConfigurationContainsCanonicalWordsGuidance_RT37_2() throws {
        let discussion = try XCTUnwrap(Transcribe.configuration.discussion)

        XCTAssertTrue(discussion.contains("transcribe words INPUT --output OUTPUT.json"))
        XCTAssertTrue(discussion.contains("transcription[].tokens[].timestamps"))
        XCTAssertTrue(discussion.contains("writes JSON to the output file, not to standard output"))
    }

    func testCanonicalDocumentDefinesEveryHelpKey_RT37_3() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let helpURL = repositoryRoot.appendingPathComponent("docs/transcribe-help.md")
        let document = try String(contentsOf: helpURL, encoding: .utf8)
        let sections = HelpText.parse(document)

        for key in HelpText.Key.allCases {
            XCTAssertFalse(
                sections[key.rawValue, default: ""].isEmpty, "Missing help section: \(key.rawValue)"
            )
        }
    }

    func testLegacyRootHelpIsNotRoutedToSummarize_RT37_4() {
        XCTAssertEqual(Transcribe.legacyArguments(["--help"]), ["--help"])
        XCTAssertEqual(Transcribe.legacyArguments(["-h"]), ["-h"])
        XCTAssertEqual(Transcribe.legacyArguments(["--version"]), ["--version"])
        XCTAssertEqual(
            Transcribe.legacyArguments(["words", "recording.m4a"]), ["words", "recording.m4a"])
        XCTAssertEqual(
            Transcribe.legacyArguments(["recording.m4a"]), ["summarize", "recording.m4a"])
    }
}
