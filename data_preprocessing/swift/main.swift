import Foundation

guard CommandLine.arguments.count == 2 else {
    print("Usage: remove_background <path to Outfield_Players>")
    exit(1)
}

let remover = BackgroundRemover(
    imageDirectory: URL(fileURLWithPath: CommandLine.arguments[1])
)

let result = remover.removeAllBackgrounds()

print("\nProcessing complete.")
print("Successfully processed: \(result.processed)")
print("Failed: \(result.failed)")

exit(result.failed == 0 ? 0 : 2)