provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  use_cli         = true
}

variable "subscription_id" {
  type        = string
  description = "Azure Subscription ID"
}

variable "ssh_public_key" {
  type        = string
  description = "SSH Public Key for the persistent VM"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH into VM"
  type        = string
  default     = "0.0.0.0/0"
}

variable "allowed_ip_for_tf_serving" {
  description = "CIDR allowed to access TF Serving"
  type        = string
  default     = "0.0.0.0/0"
}

variable "rg_name" {
  type        = string
  description = "Resource Group Name"
}

variable "vnet_name" {
  type        = string
  description = "Virtual Network Name"
}

variable "vm_name" {
  type        = string
  description = "Virtual Machine Name"
}

# Static suffix so names remain predictable
resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.rg_name
  location = "UK South"
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = var.vnet_name
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

# Subnet
resource "azurerm_subnet" "subnet" {
  name                 = "test-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "nsg" {
  name                = "test-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "Allow-SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.allowed_ssh_cidr
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-TFServing"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8501"
    source_address_prefix      = var.allowed_ip_for_tf_serving
    destination_address_prefix = "*"
  }
  
  security_rule {
    name                       = "Allow-TFServing-8502"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8502"
    source_address_prefix      = var.allowed_ip_for_tf_serving
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-Prometheus"
    priority                   = 1004
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9090"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-NodeExporter"
    priority                   = 1005
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9100"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-cAdvisor"
    priority                   = 1006
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9323"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-Grafana"
    priority                   = 1007
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "3000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-TokenGenerator"
    priority                   = 1008
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8082"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "Allow-TokenValidator"
    priority                   = 1009
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8083"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  } 

  security_rule {
    name                       = "Allow-wati"
    priority                   = 1010
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8520"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Public IP (persistent for primary VM)
resource "azurerm_public_ip" "pip_primary" {
  name                = "${var.vm_name}-pip"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Static"
  sku                 = "Standard"
  domain_name_label   = "${var.vm_name}-${random_string.suffix.result}"
}

# NIC for primary VM
resource "azurerm_network_interface" "nic_primary" {
  name                = "${var.vm_name}-nic"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pip_primary.id
  }
}

# Associate NSG to NIC
resource "azurerm_network_interface_security_group_association" "nsg_assoc_primary" {
  network_interface_id      = azurerm_network_interface.nic_primary.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

# Persistent primary VM
resource "azurerm_linux_virtual_machine" "primary_vm" {
  name                = var.vm_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = "Standard_B1s"
  admin_username      = "azureuser"
  network_interface_ids = [
    azurerm_network_interface.nic_primary.id,
  ]

  admin_ssh_key {
    username   = "azureuser"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
}

# ========================
# ✅ Outputs
# ========================

output "primary_vm_public_ip" {
  value       = azurerm_public_ip.pip_primary.ip_address
  description = "Public IP of the persistent primary VM"
}

output "public_ip" {
  value       = azurerm_public_ip.pip_primary.ip_address
  description = "Alias output for Build Pipeline compatibility"
}

output "resource_group_name" {
  value       = azurerm_resource_group.rg.name
  description = "Name of the shared Resource Group"
}

output "vnet_name" {
  value       = azurerm_virtual_network.vnet.name
  description = "Name of the shared Virtual Network"
}

output "subnet_id" {
  value       = azurerm_subnet.subnet.id
  description = "ID of the shared Subnet"
}

output "nsg_name" {
  value       = azurerm_network_security_group.nsg.name
  description = "Name of the shared Network Security Group"
}
