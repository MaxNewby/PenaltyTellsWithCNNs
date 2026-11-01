import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

final class BackgroundRemover {

    let imageDirectory: URL
    private let fileManager = FileManager.default
    private let context = CIContext()
    private let supportedExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "tiff", "tif"
    ]

    init(imageDirectory: URL) {
        self.imageDirectory = imageDirectory.standardizedFileURL
    }

    // MARK: - Single image

    func removeBackground(from inputURL: URL, to outputURL: URL) throws {

        let handler = VNImageRequestHandler(url: inputURL, options: [:])
        let request = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([request])

        guard let result = request.results?.first else {
            throw NSError(
                domain: "BackgroundRemoval", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No foreground detected"]
            )
        }

        // Full frame kept (not cropped), so the output lines up with the raw image
        let maskedBuffer = try result.generateMaskedImage(
            ofInstances: result.allInstances,
            from: handler,
            croppedToInstancesExtent: false
        )

        let ciImage = CIImage(cvPixelBuffer: maskedBuffer)

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw NSError(
                domain: "BackgroundRemoval", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Could not create output image"]
            )
        }

        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw NSError(
                domain: "BackgroundRemoval", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not create output file"]
            )
        }

        CGImageDestinationAddImage(destination, cgImage, nil)

        guard CGImageDestinationFinalize(destination) else {
            throw NSError(
                domain: "BackgroundRemoval", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not save output image"]
            )
        }
    }

    // MARK: - Batch

    /// Walks <nation>/<player>/raw and writes each result to
    /// <nation>/<player>/preprocessing/foreground/<same name>.png
    @discardableResult
    func removeAllBackgrounds() -> (processed: Int, failed: Int) {
        var processed = 0
        var failed = 0

        for nationDir in subdirectories(of: imageDirectory) {
            for playerDir in subdirectories(of: nationDir) {

                let rawDir = playerDir.appendingPathComponent("raw")
                guard fileManager.fileExists(atPath: rawDir.path) else { continue }

                let outDir = playerDir
                    .appendingPathComponent("preprocessing")
                    .appendingPathComponent("foreground")

                do {
                    try fileManager.createDirectory(
                        at: outDir, withIntermediateDirectories: true
                    )
                } catch {
                    print("[FAILED] could not create \(outDir.path): \(error)")
                    failed += 1
                    continue
                }

                for file in imageFiles(in: rawDir) {
                    let outputURL = outDir.appendingPathComponent(
                        file.deletingPathExtension().lastPathComponent + ".png"
                    )
                    do {
                        try removeBackground(from: file, to: outputURL)
                        processed += 1
                        print("[SUCCESS] \(nationDir.lastPathComponent)/\(playerDir.lastPathComponent)/\(file.lastPathComponent)")
                    } catch {
                        failed += 1
                        print("[FAILED] \(file.path): \(error)")
                    }
                }
            }
        }
        return (processed, failed)
    }

    // MARK: - Helpers

    private func subdirectories(of url: URL) -> [URL] {
        let items = (try? fileManager.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]          // skips .DS_Store
        )) ?? []
        return items
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.path < $1.path }
    }

    private func imageFiles(in url: URL) -> [URL] {
        let items = (try? fileManager.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return items
            .filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.path < $1.path }
    }
}