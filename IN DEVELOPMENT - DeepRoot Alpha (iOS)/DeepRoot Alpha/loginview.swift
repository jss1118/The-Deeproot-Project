// Imports
import SwiftUI
import UIKit
import CoreML
import Vision

// Startup
var logran: Bool = false

@main
struct startup: App {
    var body: some Scene {
        WindowGroup {
            LoginView(username: "", password: "")
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

    var body: some View {
        Group {
            if loggedin {
                MainView(loggedin: $loggedin, username: $username, password: $password)
            } else {
                ZStack {
                    // Background gradient
                    LinearGradient(gradient: Gradient(colors: [.blue.opacity(0.7), .green]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea()

                    VStack(spacing: 20) {
                        Text("Deeproot AI")
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
                    // Reset username and password when the view disappears
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
    @State var settingspressed: Bool = false
    
    // State variables for camera integration
    @State private var showCamera = false
    @State private var capturedImage: UIImage? = nil
    
    // Corrected crop list declaration.
    let crop_type: [String] = [
        "Apple", "Casava", "Cherry", "Chili", "Citrus", "Coffee",
        "Corn", "Cucumber", "Grape", "Guava", "Jamun", "Lemon",
        "Mango", "Peach", "Pepper", "Pomegranate", "Potato", "Rice",
        "Soybean", "Strawberry", "Sugarcane", "Tea", "Tomato"
    ]
    @State private var model: String = ""
    // State variable to store the selected crop type.
    @State private var selectedCrop: String = ""
    
    var body: some View {
        Group {
            if settingspressed {
                SettingsView()
            } else {
                ZStack {
                    // Background gradient
                    LinearGradient(gradient: Gradient(colors: [.blue.opacity(0.7), .green]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea()

                    // Top buttons
                    HStack(spacing: -10) {
                        Button(action: {
                            settingspressed.toggle()
                            print("Settings Button Pressed")
                        }) {
                            Label("Settings", systemImage: "gear")
                                .font(.system(size: 15))
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                                .frame(width: 130, height: 55)
                        }

                        Button(action: {
                            print("Recents Button Pressed")
                        }) {
                            Label("Recents", systemImage: "camera.macro")
                                .font(.system(size: 15))
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                                .frame(width: 145, height: 55)
                        }

                        Button(action: {
                            print("Forums Button Pressed")
                        }) {
                            Label("Forums", systemImage: "bubble.right")
                                .font(.system(size: 15))
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                                .frame(width: 125, height: 55)
                        }
                    }
                    .padding(.top, 700)

                    // Centered welcome text
                    VStack {
                        Text("Welcome.")
                            .font(.system(size: 35, weight: .bold, design: .default))
                            .frame(width: 190, height: 40)

                        Text("Diagnose the root of disease with the click of a button")
                            .font(.system(size: 20, weight: .bold, design: .default))
                            .frame(width: 300, height: 80)
                            .multilineTextAlignment(.center)
                            .padding(.bottom, 600)
                    }

                    // Diagnosis button and Crop type Menu in the center
                    HStack(spacing: 40) {
                        Button(action: {
                            print("Diagnosis Button Pressed")
                            showCamera = true  // Present the camera sheet
                        }) {
                            Label {
                                Text("Diagnosis")
                                    .font(.system(size: 20, weight: .bold, design: .default))
                                    .foregroundColor(.black)
                            } icon: {
                                Image(systemName: "camera")
                                    .font(.system(size: 40))
                            }
                            .padding()
                            .frame(width: 200, height: 200)
                            .background(LinearGradient(gradient: Gradient(colors: [.white, .blue]),
                                                       startPoint: .topLeading,
                                                       endPoint: .bottomTrailing)
                                            .ignoresSafeArea())
                            .clipShape(Circle())
                        }
                        .padding(.top, 600)

                        Menu {
                            // Use ForEach to create menu items for each crop type.
                            ForEach(crop_type, id: \.self) { type in
                                Button(action: {
                                    selectedCrop = type
                                    print(type)
                                    model = "model\(selectedCrop.lowercased())\(())"
                                    print(model)
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
                        .padding(.top, 600)
                    }
                    .padding(.bottom, 500)
                }
                .onAppear {
                    loggedin = true
                    print("Logged in. Username: \(username), Password: \(password)")
                }
                // Present the ImagePicker when showCamera is true
                .sheet(isPresented: $showCamera) {
                    ImagePicker(sourceType: .camera, image: $capturedImage)
                }
            }
        }
    }
}

struct SettingsView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
        }
    }
}

/// A UIViewControllerRepresentable wrapper for UIImagePickerController.
struct ImagePicker: UIViewControllerRepresentable {
    var sourceType: UIImagePickerController.SourceType = .camera
    @Binding var image: UIImage?
    
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController,
                                context: Context) { }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    // Coordinator to handle UIImagePickerController events.
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

#Preview {
    LoginView(username: "", password: "")
}

