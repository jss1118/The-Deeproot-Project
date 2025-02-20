//
//  ContentView.swift
//  Deeproot macOS
//
//  Created by Joshua Stanley on 2/20/25.
//

import SwiftUI

struct MainView: View {
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(gradient: Gradient(colors: [.white.opacity(0.7), .black.opacity(0.7)]),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
                .ignoresSafeArea()
            
            // Sidebar with NavigationView
            NavigationView {
                // Sidebar list
                List {
                    NavigationLink(destination: Text("Home View")) {
                        Label("Home", systemImage: "house")
                    }
                    NavigationLink(destination: Text("Settings View")) {
                        Label("Settings", systemImage: "gear")
                    }
                    NavigationLink(destination: Text("Profile View")) {
                        Label("Profile", systemImage: "person.crop.circle")
                    }
                }
                .listStyle(SidebarListStyle())
                .frame(minWidth: 200)
                
                // Default detail view when no selection is made
                Text("Select an item from the sidebar")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}

#Preview {
    MainView()
}

