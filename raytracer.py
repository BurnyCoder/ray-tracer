#!/usr/bin/env python3
"""
A simple raytracer implementation in Python that renders a scene with colorful light sources.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image


@dataclass
class Vector3:
    """A 3D vector class with basic operations."""
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.dot(self))

    def normalized(self):
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return self / length

    def to_tuple(self):
        return (self.x, self.y, self.z)


@dataclass
class Ray:
    """A ray with an origin and direction."""
    origin: Vector3
    direction: Vector3


@dataclass
class Material:
    """Material properties for objects in the scene."""
    color: Vector3
    ambient: float = 0.1
    diffuse: float = 0.7
    specular: float = 0.2
    shininess: float = 30
    reflection: float = 0.0


@dataclass
class Light:
    """A point light source with position and color."""
    position: Vector3
    color: Vector3
    intensity: float = 1.0


@dataclass
class Intersection:
    """Information about a ray-object intersection."""
    distance: float
    point: Vector3
    normal: Vector3
    material: Material


class Object:
    """Base class for all objects in the scene."""
    def intersect(self, ray: Ray) -> Optional[Intersection]:
        """Calculate the intersection of a ray with this object."""
        raise NotImplementedError


class Sphere(Object):
    """A sphere object with center, radius, and material."""
    def __init__(self, center: Vector3, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        # Find the nearest intersection point
        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            t = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                return None

        point = ray.origin + ray.direction * t
        normal = (point - self.center).normalized()
        
        return Intersection(t, point, normal, self.material)


class Plane(Object):
    """An infinite plane defined by a point and normal direction."""
    def __init__(self, point: Vector3, normal: Vector3, material: Material):
        self.point = point
        self.normal = normal.normalized()
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 0.001:
            return None

        t = self.normal.dot(self.point - ray.origin) / denom
        if t < 0.001:
            return None

        point = ray.origin + ray.direction * t
        return Intersection(t, point, self.normal, self.material)


class CheckeredPlane(Plane):
    """A plane with a checkered pattern material."""
    def __init__(self, point: Vector3, normal: Vector3, 
                 material1: Material, material2: Material, size: float = 1.0):
        super().__init__(point, normal, material1)
        self.material1 = material1
        self.material2 = material2
        self.size = size

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        intersection = super().intersect(ray)
        if not intersection:
            return None

        # Create checkered pattern
        x = math.floor(intersection.point.x / self.size)
        z = math.floor(intersection.point.z / self.size)
        
        if (x + z) % 2 == 0:
            intersection.material = self.material1
        else:
            intersection.material = self.material2

        return intersection


class Camera:
    """Camera for generating rays for the scene."""
    def __init__(self, position: Vector3, look_at: Vector3, up: Vector3, 
                 fov: float, aspect_ratio: float):
        self.position = position
        self.forward = (look_at - position).normalized()
        self.right = self.forward.cross(up).normalized()
        self.up = self.right.cross(self.forward)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        
        # Calculate the dimensions of the image plane
        self.h = math.tan(math.radians(fov / 2))
        self.w = self.h * aspect_ratio

    def generate_ray(self, x: float, y: float) -> Ray:
        """Generate a ray for the given normalized coordinates (x, y)."""
        # Convert from [0,1] to [-1,1] and flip y
        x = 2 * x - 1
        y = 1 - 2 * y
        
        # Calculate the direction vector
        direction = (self.forward +
                    self.right * (x * self.w) +
                    self.up * (y * self.h)).normalized()
        
        return Ray(self.position, direction)


class Scene:
    """A collection of objects and lights."""
    def __init__(self, objects: List[Object], lights: List[Light], 
                 background_color: Vector3 = Vector3(0, 0, 0)):
        self.objects = objects
        self.lights = lights
        self.background_color = background_color

    def trace_ray(self, ray: Ray, depth: int = 0, max_depth: int = 3) -> Vector3:
        """Trace a ray through the scene and return the color."""
        if depth > max_depth:
            return self.background_color

        nearest_intersection = self._find_nearest_intersection(ray)
        if not nearest_intersection:
            return self.background_color

        return self._calculate_color(ray, nearest_intersection, depth, max_depth)

    def _find_nearest_intersection(self, ray: Ray) -> Optional[Intersection]:
        """Find the nearest object intersected by the ray."""
        nearest_intersection = None
        min_distance = float('inf')

        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection and intersection.distance < min_distance:
                min_distance = intersection.distance
                nearest_intersection = intersection

        return nearest_intersection

    def _calculate_color(self, ray: Ray, intersection: Intersection, 
                         depth: int, max_depth: int) -> Vector3:
        """Calculate the color at an intersection point."""
        material = intersection.material
        point = intersection.point
        normal = intersection.normal
        
        # Initialize with ambient light
        color = material.color * material.ambient
        
        # For each light source
        for light in self.lights:
            light_dir = (light.position - point).normalized()
            
            # Shadow check
            shadow_ray = Ray(point + normal * 0.001, light_dir)
            shadow_intersection = self._find_nearest_intersection(shadow_ray)
            
            if shadow_intersection:
                light_distance = (light.position - point).length()
                if shadow_intersection.distance < light_distance:
                    continue  # Point is in shadow
            
            # Diffuse reflection
            n_dot_l = max(normal.dot(light_dir), 0)
            diffuse = material.diffuse * n_dot_l
            
            # Specular reflection
            reflect_dir = normal * (2 * n_dot_l) - light_dir
            view_dir = (ray.origin - point).normalized()
            r_dot_v = max(reflect_dir.dot(view_dir), 0)
            specular = material.specular * (r_dot_v ** material.shininess)
            
            # Add contribution from this light
            light_color = light.color * light.intensity
            color = color + light_color * (material.color * diffuse + Vector3(1, 1, 1) * specular)
        
        # Calculate reflection
        if material.reflection > 0 and depth < max_depth:
            reflect_dir = ray.direction - normal * (2 * ray.direction.dot(normal))
            reflect_ray = Ray(point + normal * 0.001, reflect_dir)
            reflect_color = self.trace_ray(reflect_ray, depth + 1, max_depth)
            color = color * (1 - material.reflection) + reflect_color * material.reflection
        
        # Clamp color components
        return Vector3(
            min(1, max(0, color.x)),
            min(1, max(0, color.y)),
            min(1, max(0, color.z))
        )


def create_test_scene() -> Scene:
    """Create a test scene with multiple colored light sources."""
    # Create materials
    red = Material(Vector3(1.0, 0.2, 0.2), reflection=0.2)
    green = Material(Vector3(0.2, 1.0, 0.2), reflection=0.2)
    blue = Material(Vector3(0.2, 0.2, 1.0), reflection=0.2)
    yellow = Material(Vector3(1.0, 1.0, 0.2), reflection=0.2)
    purple = Material(Vector3(0.8, 0.2, 0.8), reflection=0.2)
    cyan = Material(Vector3(0.2, 0.8, 0.8), reflection=0.2)
    white = Material(Vector3(0.9, 0.9, 0.9), reflection=0.5)
    black = Material(Vector3(0.1, 0.1, 0.1), reflection=0.0)
    mirror = Material(Vector3(0.9, 0.9, 0.9), ambient=0.1, diffuse=0.1, specular=0.8, reflection=0.9)
    
    # Create objects
    objects = [
        # Floor
        CheckeredPlane(Vector3(0, -3, 0), Vector3(0, 1, 0), white, black, size=2.0),
        
        # Main spheres
        Sphere(Vector3(-4, 0, -6), 2, red),
        Sphere(Vector3(0, 0, -6), 2, mirror),
        Sphere(Vector3(4, 0, -6), 2, blue),
        
        # Smaller spheres
        Sphere(Vector3(-2.5, -1.5, -3), 1, green),
        Sphere(Vector3(2.5, -1.5, -3), 1, purple),
        Sphere(Vector3(0, -1.5, -9), 1, yellow),
        Sphere(Vector3(0, -1.5, -3), 1, cyan),
    ]
    
    # Create colorful lights
    lights = [
        Light(Vector3(-6, 6, 0), Vector3(1.0, 0.5, 0.5), 1.2),  # Red light
        Light(Vector3(6, 6, 0), Vector3(0.5, 0.5, 1.0), 1.2),   # Blue light
        Light(Vector3(0, 6, 6), Vector3(0.5, 1.0, 0.5), 1.2),   # Green light
        Light(Vector3(0, 6, -10), Vector3(1.0, 1.0, 1.0), 0.8), # White light
        Light(Vector3(-6, 6, -10), Vector3(1.0, 1.0, 0.5), 1.0), # Yellow light
        Light(Vector3(6, 6, -10), Vector3(0.8, 0.5, 1.0), 1.0),  # Purple light
    ]
    
    return Scene(objects, lights, Vector3(0.05, 0.05, 0.1))


def render(scene: Scene, camera: Camera, width: int, height: int, 
           samples: int = 1, max_depth: int = 3) -> np.ndarray:
    """Render the scene and return the image as a numpy array."""
    image = np.zeros((height, width, 3))
    
    for y in range(height):
        print(f"Rendering line {y+1}/{height}")
        for x in range(width):
            color = Vector3(0, 0, 0)
            
            # Anti-aliasing with multiple samples
            for _ in range(samples):
                if samples == 1:
                    u, v = (x + 0.5) / width, (y + 0.5) / height
                else:
                    u = (x + random.random()) / width
                    v = (y + random.random()) / height
                
                ray = camera.generate_ray(u, v)
                color = color + scene.trace_ray(ray, max_depth=max_depth)
            
            color = color / samples
            
            # Gamma correction
            color.x = math.sqrt(color.x)
            color.y = math.sqrt(color.y)
            color.z = math.sqrt(color.z)
            
            # Set pixel color
            image[y, x] = [color.x, color.y, color.z]
    
    # Convert to 8-bit RGB
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def main():
    """Main function to set up and render the scene."""
    # Image dimensions
    width, height = 800, 600
    aspect_ratio = width / height
    
    # Create camera
    camera = Camera(
        position=Vector3(0, 1, 10),
        look_at=Vector3(0, 0, -5),
        up=Vector3(0, 1, 0),
        fov=60,
        aspect_ratio=aspect_ratio
    )
    
    # Create scene
    scene = create_test_scene()
    
    # Render the scene
    print("Rendering scene...")
    image_data = render(scene, camera, width, height, samples=2, max_depth=3)
    
    # Save image
    image = Image.fromarray(image_data)
    image.save("raytraced_scene.png")
    print("Image saved as 'raytraced_scene.png'")


if __name__ == "__main__":
    main() 