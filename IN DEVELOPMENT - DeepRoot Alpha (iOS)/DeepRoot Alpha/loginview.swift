import SwiftUI
import UIKit
import CoreML
import Vision
import AVFoundation
import Foundation
import Combine
import PhotosUI

// Global dictionary mapping crop names to class labels.
// Make sure the order corresponds to your model's output.
let classificationLabels: [String: [String]] = [
    "apple": ["Apple__black_rot", "Apple__healthy", "Apple__rust", "Apple__scab"],
    "casava": ["Cassava__bacterial_blight", "Cassava__brown_streak_disease", "Cassava__green_mottle", "Cassava__healthy", "Cassava__mosaic_disease"],
    "cherry": ["Cherry__healthy", "Cherry__powdery_mildew"],
    "chili": ["Chili__healthy", "Chili__leaf curl", "Chili__leaf spot", "Chili__whitefly", "Chili__yellowish"],
    "citrus": ["Black spot", "canker", "greening", "healthy"],
    "coffee": ["Coffee__cercospora_leaf_spot", "Coffee__healthy", "Coffee__red_spider_mite", "Coffee__rust"],
    "corn": ["Corn__common_rust", "Corn__gray_leaf_spot", "Corn__healthy", "Corn__northern_leaf_blight"],
    "cucumber": ["Cucumber__diseased", "Cucumber__healthy"],
    "grape": ["Grape__black_measles", "Grape__black_rot", "Grape__healthy", "Grape__leaf_blight_(isariopsis_leaf_spot)"],
    "guava": ["Gauva__diseased", "Gauva__healthy"],
    "jamun": ["Jamun__diseased", "Jamun__healthy"],
    "lemon": ["Lemon__diseased", "Lemon__healthy"],
    "mango": ["Mango__diseased", "Mango__healthy"],
    "peach": ["Peach__bacterial_spot", "Peach__healthy"],
    "pepper": ["Pepper_bell__bacterial_spot", "Pepper_bell__healthy"],
    "pomegranate": ["Pomegranate__diseased", "Pomegranate__healthy"],
    "potato": ["Potato__early_blight", "Potato__healthy", "Potato__late_blight"],
    "rice": ["Rice__brown_spot", "Rice__healthy", "Rice__hispa", "Rice__leaf_blast", "Rice__neck_blast"],
    "soybean": ["Soybean__bacterial_blight", "Soybean__caterpillar", "Soybean__diabrotica_speciosa", "Soybean__downy_mildew", "Soybean__healthy", "Soybean__mosaic_virus", "Soybean__powdery_mildew", "Soybean__rust", "Soybean__southern_blight"],
    "strawberry": ["Strawberry___leaf_scorch", "Strawberry__healthy"],
    "sugarcane": ["Sugarcane__bacterial_blight", "Sugarcane__healthy", "Sugarcane__red_rot", "Sugarcane__red_stripe", "Sugarcane__rust"],
    "tea": ["Tea__algal_leaf", "Tea__anthracnose", "Tea__bird_eye_spot", "Tea__brown_blight", "Tea__healthy"],
    "tomato": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]
]

// MARK: - Helper: Convert a CIImage to MLMultiArray
func convertCIImageToMLMultiArray(_ image: CIImage, targetSize: CGSize = CGSize(width: 128, height: 128)) -> MLMultiArray? {
    let context = CIContext(options: nil)
    guard let cgImage = context.createCGImage(image, from: image.extent) else { return nil }
    UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
    UIImage(cgImage: cgImage).draw(in: CGRect(origin: .zero, size: targetSize))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    guard let uiImage = resizedImage,
          let pixelBuffer = uiImage.pixelBuffer(width: Int(targetSize.width), height: Int(targetSize.height)) else { return nil }
    guard let multiArray = try? MLMultiArray(shape: [1, 128, 128, 3], dataType: .double) else { return nil }
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    for y in 0..<height {
        for x in 0..<width {
            let pixel = baseAddress.advanced(by: y * bytesPerRow + x * 4)
            let b = Double(pixel.load(as: UInt8.self))
            let g = Double(pixel.advanced(by: 1).load(as: UInt8.self))
            let r = Double(pixel.advanced(by: 2).load(as: UInt8.self))
            let normalizedR = r / 255.0
            let normalizedG = g / 255.0
            let normalizedB = b / 255.0
            let index = ((0 * height + y) * width + x) * 3
            multiArray[index + 0] = NSNumber(value: normalizedR)
            multiArray[index + 1] = NSNumber(value: normalizedG)
            multiArray[index + 2] = NSNumber(value: normalizedB)
        }
    }
    return multiArray
}

// MARK: - UIImage Extension to create a CVPixelBuffer
extension UIImage {
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        else { return nil }
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        UIGraphicsPopContext()
        return buffer
    }
}

// MARK: - Camera ViewModel
class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    // MARK: - Properties
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private var requests = [VNRequest]()
    @Published var detections: [VNRecognizedObjectObservation] = []
    @Published var detectionLabels: [UUID: String] = [:]
    @Published var selectedCrop: String = ""
    private var currentBuffer: CVPixelBuffer?
    private let videoQueue = DispatchQueue(label: "videoQueue")
    
    override init() {
        super.init()
        setupSession()
        setupVision()  // YOLO-based object detection
    }
    
    private func setupSession() {
        session.sessionPreset = .high
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Could not create video device input.")
            return
        }
        session.addInput(input)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                connection.videoRotationAngle = 0
            } else {
                connection.videoOrientation = .portrait
            }
        }
    }
    
    private func setupVision() {
        do {
            let config = MLModelConfiguration()
            let yoloModel = try yolocore(configuration: config)
            let visionModel = try VNCoreMLModel(for: yoloModel.model)
            let objectRecognition = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
                guard let self = self else { return }
                self.visionRequestDidComplete(request: request, error: error)
            }
            objectRecognition.imageCropAndScaleOption = .centerCrop
            requests = [objectRecognition]
        } catch {
            print("Error setting up the CoreML detection model: \(error.localizedDescription)")
        }
    }
    
    func startSession() {
        session.startRunning()
    }
    
    func stopSession() {
        session.stopRunning()
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !requests.isEmpty,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        self.currentBuffer = pixelBuffer
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform(requests)
        } catch {
            print("Failed to perform detection: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Vision Completion Handler
    private func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let error = error {
            print("Vision error: \(error.localizedDescription)")
            return
        }
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            print("No recognized objects returned.")
            return
        }
        let confidenceThreshold: VNConfidence = 0.25
        let filtered = observations.filter { obs in
            guard let bestLabel = obs.labels.first else { return false }
            return bestLabel.confidence >= confidenceThreshold
        }
        print("Total detections: \(observations.count). Kept after filtering: \(filtered.count)")
        for detection in filtered {
            if let topLabel = detection.labels.first {
                print("Detected: \(topLabel.identifier) [confidence: \(topLabel.confidence)] with bbox: \(detection.boundingBox)")
            }
        }
        DispatchQueue.main.async {
            self.detections = filtered
        }
        if let buffer = self.currentBuffer {
            for detection in filtered {
                self.classifyDetection(detection: detection, pixelBuffer: buffer)
            }
        }
    }
    
    /// Modified classifyDetection using multi-array conversion and numeric output mapping.
    private func classifyDetection(detection: VNRecognizedObjectObservation, pixelBuffer: CVPixelBuffer) {
        guard !selectedCrop.isEmpty else {
            print("No crop selected – skipping classifier step.")
            return
        }
        let classifierModelName = "model" + selectedCrop.lowercased()
        guard let modelURL = Bundle.main.url(forResource: classifierModelName, withExtension: "mlmodelc") else {
            print("Classifier model \(classifierModelName) not found in bundle.")
            return
        }
        guard let mlModel = try? MLModel(contentsOf: modelURL) else {
            print("Could not load classifier model \(classifierModelName).")
            return
        }
        // Crop the detected region.
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let imageExtent = ciImage.extent
        let bbox = detection.boundingBox
        let cropRect = CGRect(
            x: bbox.origin.x * imageExtent.width,
            y: (1 - bbox.origin.y - bbox.height) * imageExtent.height,
            width: bbox.width * imageExtent.width,
            height: bbox.height * imageExtent.height
        )
        let croppedImage = ciImage.cropped(to: cropRect)
        // Convert the cropped image to an MLMultiArray.
        guard let inputArray = convertCIImageToMLMultiArray(croppedImage, targetSize: CGSize(width: 128, height: 128)) else {
            print("Failed to convert cropped image to MLMultiArray")
            return
        }
        // Prepare the feature provider.
        let inputName = mlModel.modelDescription.inputDescriptionsByName.keys.first ?? "input"
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: [inputName: inputArray]) else {
            print("Failed to create feature provider for classifier model \(classifierModelName)")
            return
        }
        guard let prediction = try? mlModel.prediction(from: inputProvider) else {
            print("Failed to get prediction from classifier model \(classifierModelName)")
            return
        }
        // Instead of expecting a string output, get the raw multi-array output.
        if let outputMultiArray = prediction.featureValue(for: "Identity")?.multiArrayValue {
            let count = outputMultiArray.count
            var maxValue = -Double.infinity
            var maxIndex = 0
            for i in 0..<count {
                let val = outputMultiArray[i].doubleValue
                if val > maxValue {
                    maxValue = val
                    maxIndex = i
                }
            }
            // Use the dictionary mapping to get the label.
            if let labels = classificationLabels[selectedCrop.lowercased()] {
                if maxIndex < labels.count {
                    let predictedLabel = labels[maxIndex]
                    print("Crop: \(selectedCrop) → Classified as: \(predictedLabel) with confidence \(maxValue)")
                    DispatchQueue.main.async {
                        self.detectionLabels[detection.uuid] = "\(predictedLabel) (\(String(format: "%.2f", maxValue)))"
                    }
                } else {
                    print("Predicted index \(maxIndex) out of range for crop \(selectedCrop)")
                }
            } else {
                print("No label mapping for selected crop: \(selectedCrop)")
            }
        } else {
            print("No multi-array output for classifier model \(classifierModelName)")
        }
    }
}

// MARK: - Camera Preview
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var onPreviewLayerReady: ((AVCaptureVideoPreviewLayer) -> Void)? = nil
    
    class VideoPreviewView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
    
    func makeUIView(context: Context) -> VideoPreviewView {
        let view = VideoPreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        DispatchQueue.main.async {
            self.onPreviewLayerReady?(view.videoPreviewLayer)
        }
        return view
    }
    
    func updateUIView(_ uiView: VideoPreviewView, context: Context) {}
}

// MARK: - Camera View
struct CameraView: View {
    @ObservedObject var cameraVM: CameraViewModel
    @State private var previewLayer: AVCaptureVideoPreviewLayer?
    
    var body: some View {
        GeometryReader { _ in
            ZStack {
                CameraPreview(session: cameraVM.session) { layer in
                    previewLayer = layer
                }
                .ignoresSafeArea()
                ForEach(cameraVM.detections, id: \.uuid) { detection in
                    if let layer = previewLayer {
                        let convertedRect: CGRect = {
                            var normRect = detection.boundingBox
                            normRect.origin.y = 1 - normRect.origin.y - normRect.height
                            return layer.layerRectConverted(fromMetadataOutputRect: normRect)
                        }()
                        ZStack {
                            Rectangle()
                                .stroke(Color.red, lineWidth: 2)
                                .frame(width: convertedRect.width, height: convertedRect.height)
                                .position(x: convertedRect.midX, y: convertedRect.midY)
                            if let labelText = cameraVM.detectionLabels[detection.uuid] {
                                Text(labelText)
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .padding(4)
                                    .background(Color.black.opacity(0.7))
                                    .cornerRadius(4)
                                    .position(x: convertedRect.minX + 60, y: convertedRect.minY - 10)
                            }
                        }
                    }
                }
            }
            .onAppear {
                cameraVM.startSession()
            }
            .onDisappear {
                cameraVM.stopSession()
            }
        }
    }
}

// MARK: - Login & Main Views
var logran: Bool = false

@main
struct startup: App {
    @StateObject private var cameraVM = CameraViewModel()
    var body: some Scene {
        WindowGroup {
            LoginView(username: "", password: "")
                .environmentObject(cameraVM)
        }
    }
}

func getid() {
    if !logran {
        if let idForVendor = UIDevice.current.identifierForVendor?.uuidString {
            print("Vendor Identifier: \(idForVendor)")
            logran = true
        }
    }
}

struct LoginView: View {
    @State public var username: String
    @State public var password: String
    @State private var loggedin = false
    @EnvironmentObject var cameraVM: CameraViewModel
    
    var body: some View {
        Group {
            if loggedin {
                NavigationView {
                    MainView(loggedin: $loggedin, username: $username, password: $password)
                        .navigationBarHidden(true)
                }
            } else {
                ZStack {
                    LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea()
                    VStack(spacing: 20) {
                        Text("Deeproot Alpha")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        Text("Login")
                            .font(.title)
                            .fontWeight(.medium)
                            .foregroundColor(.white)
                        VStack(spacing: 15) {
                            TextField("Username", text: $username)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding()
                                .textInputAutocapitalization(.never)
                                .disableAutocorrection(true)
                            SecureField("Password", text: $password)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding()
                                .textInputAutocapitalization(.never)
                                .disableAutocorrection(true)
                        }
                        .background(RoundedRectangle(cornerRadius: 15)
                            .fill(Color.white.opacity(0.2)))
                        .padding()
                        Button(action: {
                            loggedin = true
                        }) {
                            Text("Login")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(Color.blue)
                                .cornerRadius(10)
                                .padding(.horizontal)
                        }
                    }
                    .padding()
                }
                .onDisappear {
                    username = ""
                    password = ""
                }
            }
        }
    }
}

struct MainView: View {
    @Binding var loggedin: Bool
    @Binding var username: String
    @Binding var password: String
    @EnvironmentObject var cameraVM: CameraViewModel
    @State private var showCamera = false
    let crop_type: [String] = [
        "Apple", "Casava", "Cherry", "Chili", "Citrus", "Coffee",
        "Corn", "Cucumber", "Grape", "Guava", "Jamun", "Lemon",
        "Mango", "Peach", "Pepper", "Pomegranate", "Potato", "Rice",
        "Soybean", "Strawberry", "Sugarcane", "Tea", "Tomato"
    ]
    @State private var selectedCrop: String = ""
    
    var body: some View {
        ZStack {
            LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
                .ignoresSafeArea()
            VStack {
                Spacer()
                VStack(spacing: 200) {
                    VStack {
                        Text("Welcome.")
                            .font(.system(size: 35, weight: .bold))
                            .frame(width: 190, height: 40)
                            .padding()
                        Text("Diagnose the root of disease with the click of a button")
                            .font(.system(size: 20, weight: .bold))
                            .frame(width: 300, height: 80)
                            .multilineTextAlignment(.center)
                    }
                    HStack(spacing: 40) {
                        Button(action: {
                            showCamera = true
                        }) {
                            Label {
                                Text("Diagnosis")
                                    .font(.system(size: 20, weight: .bold))
                                    .foregroundColor(.black)
                            } icon: {
                                Image(systemName: "camera")
                                    .font(.system(size: 40))
                            }
                            .frame(width: 200, height: 200)
                            .background(LinearGradient(gradient: Gradient(colors: [.white, .blue]),
                                                       startPoint: .topLeading,
                                                       endPoint: .bottomTrailing))
                            .clipShape(Circle())
                        }
                        .sheet(isPresented: $showCamera) {
                            CameraView(cameraVM: cameraVM)
                        }
                        .padding(.bottom, 100)
                        Menu {
                            ForEach(crop_type, id: \.self) { type in
                                Button(action: {
                                    selectedCrop = type
                                    print("Selected crop: \(type)")
                                    cameraVM.selectedCrop = type
                                }) {
                                    Text(type)
                                }
                            }
                        } label: {
                            Text(selectedCrop.isEmpty ? "Crop Type" : selectedCrop)
                                .fontWeight(.bold)
                                .padding()
                                .background(Color.white)
                                .cornerRadius(5)
                        }
                        .padding(.bottom, 100)
                    }
                }
                Spacer()
                HStack(spacing: 50) {
                    NavigationLink(destination: SettingsView()) {
                        Label("Settings", systemImage: "gear")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    NavigationLink(destination: DeveloperView()) {
                        Label("Developer", systemImage: "hammer.circle")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                }
                .padding(.bottom, 30)
            }
        }
        .onAppear {
            print("Logged in. Username: \(username), Password: \(password)")
        }
    }
}

struct SettingsView: View {
    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                               startPoint: .topLeading,
                               endPoint: .bottomTrailing)
                    .ignoresSafeArea()
                VStack(alignment: .leading, spacing: 20) {
                    Button(action: { print("Model btn pressed") }) {
                        Text("Models")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    Button(action: { print("Theme btn pressed") }) {
                        Text("Theme")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    Button(action: { print("Beta btn pressed") }) {
                        Text("Beta Development")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    Spacer()
                    Image("logotransparent")
                        .resizable()
                        .frame(width: 400, height: 400)
                        .aspectRatio(contentMode: .fill)
                        .padding()
                    Spacer()
                }
            }
            .navigationBarTitle("Settings", displayMode: .inline)
        }
    }
}

struct DeveloperView: View {
    @State private var devkey: String = ""
    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                               startPoint: .topLeading,
                               endPoint: .bottomTrailing)
                    .ignoresSafeArea()
                VStack(spacing: 15) {
                    Text("Enter your developer key")
                        .font(.title)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    Text("Get a developer key")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .frame(width: 400, height: 100, alignment: .center)
                        .padding(.bottom, 20)
                    TextField("Developer key", text: $devkey)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .padding()
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                }
                .padding()
            }
        }
    }
}

